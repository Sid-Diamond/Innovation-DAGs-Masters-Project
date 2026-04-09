import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from scipy.special import beta as beta_func
from scipy.stats import chisquare
from typing import Dict, Tuple, List
import time
from scipy.stats import kstest        
import matplotlib.patches as mpatches
        

class DirectedHomophilicNetwork:
    """Optimized directed network with homophilic preferential attachment."""
    
    def __init__(self, n0: int, n_nodes: int, m_edges: int, h: float, f_a: float, mu_a: float, mu_b: float):
        # Network parameters
        self.n0, self.n_nodes, self.m_edges = n0, n_nodes, m_edges
        self.h, self.f_a, self.f_b = h, f_a, 1 - f_a
        self.mu = {'a': mu_a, 'b': mu_b}
        
        # Computed parameters
        self.lambda_a = h * f_a + (1 - f_a) * (1 - h)
        self.lambda_b = h * (1 - f_a) + (1 - h) * f_a
        self.lambda_ = {'a': self.lambda_a, 'b': self.lambda_b}
        
        # Network state
        self.graph = None
        self.node_types = None
        self.edge_evolution = []
        self.g_a, self.g_b, self.g_b_asymptotic, self.Z_factor, self.Z_tilde = None, None, None, None, None
        
        # Cached state for speed
        self.in_degrees = None
        self.in_edges_a_count = 0
        self.in_edges_b_count = 0
    
    def _get_params(self, node_type: str) -> Dict[str, float]:
        """Get all distribution parameters for a node type."""
        lambda_x = self.lambda_[node_type]
        mu_x = self.mu[node_type]
        
        alpha = mu_x / lambda_x
        gamma = 1 + 1 / (self.Z_tilde * lambda_x)
        b0 = 1 / (1 + mu_x * self.Z_tilde)
        A = b0 * gamma_func(alpha + gamma) / gamma_func(alpha)
        
        return {'alpha': alpha, 'gamma': gamma, 'b0': b0, 'A': A}

    def _compute_theoretical_params(self, g_a: float, g_b: float):
        """Compute Z̃ using asymptotic g values."""
        self.g_a = g_a
        self.g_b_asymptotic = g_b
        self.g_b = self.m_edges - g_a  # Theoretical constraint
        self.Z_factor = (self.g_a * self.lambda_a + self.g_b * self.lambda_b + 
                        self.f_a * self.mu['a'] + self.f_b * self.mu['b'])
        self.Z_tilde = self.m_edges / self.Z_factor

    def assign_node_type(self) -> str:
        """Randomly assign node type based on f_a."""
        return 'a' if np.random.rand() < self.f_a else 'b'
        
    def homophilic_preferential_attachment(self, n_nodes_so_far: int) -> np.ndarray:
        """Select m_edges targets using cached in-degrees."""
        in_deg = self.in_degrees[:n_nodes_so_far]
        types = self.node_types[:n_nodes_so_far]
        
        lambda_vals = np.where(types == 0, self.lambda_a, self.lambda_b)
        mu_vals = np.where(types == 0, self.mu['a'], self.mu['b'])
        
        probs = lambda_vals * in_deg + mu_vals
        probs = probs / probs.sum()
        
        return np.random.choice(n_nodes_so_far, size=self.m_edges, p=probs, replace=False)
    
    def generate_network(self):
        """Generate network with cached state."""
        total_nodes = self.n0 + self.n_nodes
        
        # Pre-allocate arrays
        self.node_types = np.empty(total_nodes, dtype=np.int8)
        self.in_degrees = np.zeros(total_nodes, dtype=np.int32)
        self.graph = nx.DiGraph()
        
        # Initialize nodes
        for i in range(self.n0):
            self.node_types[i] = 0 if self.assign_node_type() == 'a' else 1
            self.graph.add_node(i)
        
        # Initial random edges
        for source in range(self.n0):
            targets = np.random.choice([t for t in range(self.n0) if t != source], 
                                       size=self.m_edges, replace=False)
            self.graph.add_edges_from((source, int(t)) for t in targets)
            self.in_degrees[targets] += 1
            
            for t in targets:
                if self.node_types[t] == 0:
                    self.in_edges_a_count += 1
                else:
                    self.in_edges_b_count += 1
        
        # Add nodes with preferential attachment
        for new_node in range(self.n0, total_nodes):
            self.node_types[new_node] = 0 if self.assign_node_type() == 'a' else 1
            self.graph.add_node(new_node)
            
            targets = self.homophilic_preferential_attachment(new_node)
            self.graph.add_edges_from((new_node, int(t)) for t in targets)
            
            self.in_degrees[targets] += 1
            
            for t in targets:
                if self.node_types[t] == 0:
                    self.in_edges_a_count += 1
                else:
                    self.in_edges_b_count += 1
            
            # Track evolution
            if (new_node - self.n0) % 100 == 0 or new_node == total_nodes - 1:
                self.edge_evolution.append({
                    't': new_node,
                    'in_edges_a': self.in_edges_a_count,
                    'in_edges_b': self.in_edges_b_count
                })
        
        self._fit_asymptotes()
    
    def _fit_asymptotes(self, fraction: float = 0.05):
        """Fit asymptotic g values from evolution data."""
        mean_deg_a = np.array([d['in_edges_a']/d['t'] for d in self.edge_evolution])
        mean_deg_b = np.array([d['in_edges_b']/d['t'] for d in self.edge_evolution])
        
        n_tail = max(1, int(len(mean_deg_a) * fraction))
        g_a = mean_deg_a[-n_tail:].mean()
        g_b = mean_deg_b[-n_tail:].mean()
        
        self._compute_theoretical_params(g_a, g_b)
    
    def theoretical_distribution(self, k, node_type: str):
        """
        Theoretical in-degree distribution with analytic continuation.
        Accepts scalar or array k (integer or real).
        Form: p(k) = b_0 * B(k, alpha + gamma) / B(k, alpha) for k > 0
        """
        
        params = self._get_params(node_type)
        k = np.atleast_1d(k)
        result = np.zeros_like(k, dtype=float)
        
        zero_mask = (k == 0)
        pos_mask = (k > 0)
        
        result[zero_mask] = params['b0']
        if np.any(pos_mask):
            k_pos = k[pos_mask]
            result[pos_mask] = params['b0'] * beta_func(k_pos, params['alpha'] + params['gamma']) / beta_func(k_pos, params['alpha'])
        
        return result.item() if result.shape == (1,) else result

    def _get_degrees(self, node_type: str) -> List[int]:
        """Get in-degrees for nodes of specified type."""
        type_val = 0 if node_type == 'a' else 1
        return [self.graph.in_degree(n) for n in self.graph.nodes() 
                if self.node_types[n] == type_val]
    
    def logarithmic_binning(self, degrees: np.ndarray, bin_factor: float = 1.02) -> Tuple[np.ndarray, np.ndarray]:
        """Create logarithmic bins for degree distribution."""
        if len(degrees) == 0:
            return np.array([]), np.array([])
        
        degrees = np.array(degrees)
        max_degree, n_total = np.max(degrees), len(degrees)
        
        # Create bins
        bins, current = [0], 1
        while current <= max_degree:
            bins.append(int(current))
            current *= bin_factor
        bins.append(int(max_degree) + 1)
        bins = sorted(set(bins))
        
        # Compute bin statistics
        bin_centers, probabilities = [], []
        for i in range(len(bins) - 1):
            kmin, kmax = bins[i], bins[i + 1] - 1
            count = np.sum((degrees >= kmin) & (degrees <= kmax))
            
            if count > 0:
                center = np.sqrt(0.5 * kmax) if kmin == 0 and kmax > 0 else (0.0 if kmin == 0 else np.sqrt(kmin * kmax))
                bin_centers.append(center)
                probabilities.append(count / n_total)
        
        return np.array(bin_centers), np.array(probabilities)

    def plot_degree_distributions(self, figsize: Tuple = (15, 6), discretisations: int = 10**5):
            """
            Plot in-degree distributions with theoretical curves.
            
            Parameters:
            -----------
            discretisations : int
                0 = integer k values only, >0 = number of continuous points for smooth curve
            """
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
                in_degrees = self._get_degrees(node_type)
                if len(in_degrees) == 0:
                    continue
                    
                ax = axes[idx]
                
                # Empirical data
                bin_centers, probs = self.logarithmic_binning(in_degrees)
                ax.scatter(bin_centers, probs, s=50, alpha=0.7, color=color,
                        edgecolors='black', linewidths=0.5, label='Simulation', zorder=3)
                
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)
                
                # Theoretical curve
                k_max = int(np.max(in_degrees))
                k_range = np.arange(0, k_max + 1) if discretisations == 0 else np.concatenate([[0], np.linspace(0.01, k_max, discretisations)])
                theo_probs = self.theoretical_distribution(k_range, node_type)
                mask = theo_probs > 0
                
                ax.plot(k_range[mask], theo_probs[mask], '-', linewidth=2.5, color='dark'+color, 
                        alpha=0.85, label='Theory', zorder=2)
                
                # Formatting
                ax.set_xscale('symlog', linthresh=0.1)
                ax.set_yscale('log')
                ax.set_xlim(left=-0.05)
                ax.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=13)
                ax.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=13)
                ax.set_title(f'Type "{node_type}" (n={len(in_degrees)})', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9, loc='best', framealpha=0.9)
                ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
            
            fig.suptitle(f'Directed Homophilic Network: N={self.n0 + self.n_nodes:,}, '
                        f'm={self.m_edges}, h={self.h}, f_a={self.f_a}',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig

    def plot_in_edge_asymptotes(self, figsize: Tuple = (10, 6)):
        """Plot mean in-edge density with asymptotic fits."""
        times = np.array([d['t'] for d in self.edge_evolution])
        mean_deg_a = np.array([d['in_edges_a']/d['t'] for d in self.edge_evolution])
        mean_deg_b = np.array([d['in_edges_b']/d['t'] for d in self.edge_evolution])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for mean_deg, type_name, color in [(mean_deg_a, 'a', 'red'), (mean_deg_b, 'b', 'blue')]:
            ax.plot(times, mean_deg, label=f"Type '{type_name}' (data)", color=color)
            asymptote = self.g_a if type_name == 'a' else self.g_b_asymptotic
            ax.axhline(asymptote, linestyle='--', alpha=0.7, color='dark'+color,
                      label=f"Type '{type_name}' asymptote = {asymptote:.3f}")
        
        ax.set_xlabel("t (number of nodes)")
        ax.set_ylabel("Mean in-degree")
        ax.set_title("Asymptotic In-Edge Density")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_A_values(self, max_k: int = 25):
        """Plot normalization constant A(k) for both types."""
        k_values = np.arange(0, max_k + 1)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
            params = self._get_params(node_type)
            
            # Compute A(k) by brute force
            A_values = []
            for k in k_values:
                product = np.prod([(params['alpha'] + i) / (params['alpha'] + params['gamma'] + i) 
                                  for i in range(k)]) if k > 0 else 1.0
                gamma_ratio = gamma_func(k + params['alpha'] + params['gamma']) / gamma_func(k + params['alpha'])
                A_values.append(params['b0'] * product * gamma_ratio)
            
            ax = axes[idx]
            ax.plot(k_values, A_values, 'o-', color=color, linewidth=2, 
                   markersize=4, alpha=0.7, label='A(k) computed')
            ax.axhline(params['A'], linestyle='--', color='black', linewidth=2, alpha=0.7,
                      label=f"b₀·Γ(α+γ)/Γ(α) = {params['A']:.2f}")
            
            ax.set_xlabel('k', fontsize=13)
            ax.set_ylabel('A(k)', fontsize=13)
            ax.set_title(f"Type '{node_type}' Normalization", fontsize=13, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def theoretical_cdf(self, k, node_type: str):
        """Theoretical CDF via summation of PMF."""
        k = np.atleast_1d(k)
        cdf = np.zeros_like(k, dtype=float)
        
        for i, k_val in enumerate(k):
            if k_val < 0:
                cdf[i] = 0.0
            else:
                k_range = np.arange(0, int(np.floor(k_val)) + 1)
                cdf[i] = np.sum(self.theoretical_distribution(k_range, node_type))
        
        return cdf.item() if cdf.shape == (1,) else cdf

    def ks_test(self, node_type='b', min_degree=0, min_bin_count=10):
        """
        KS test with manual D statistic and scipy's p-value calculation.
        """
        from scipy.stats import ksone
        
        degrees = np.array(self._get_degrees(node_type))
        max_degree = int(np.max(degrees))
        
        # Find valid range
        k_values = np.arange(min_degree, max_degree + 1)
        observed_counts = np.array([np.sum(degrees == k) for k in k_values])
        valid_mask = observed_counts >= min_bin_count
        
        if not np.any(valid_mask):
            print(f"No bins with >= {min_bin_count} observations!")
            return None, None
        
        max_k = k_values[np.where(valid_mask)[0][-1]]
        
        # Filter degrees to test range
        degrees_filtered = degrees[(degrees >= min_degree) & (degrees <= max_k)]
        percent_used = 100 * len(degrees_filtered) / len(degrees)
        n = len(degrees_filtered)
        
        # Compute manual D statistic (our correct one)
        unique_degrees = np.sort(np.unique(degrees_filtered))
        empirical_cdf = np.array([np.sum(degrees_filtered <= k) / n for k in unique_degrees])
        theoretical_cdf_vals = np.array([self.theoretical_cdf(k, node_type) for k in unique_degrees])
        discrepancies = np.abs(empirical_cdf - theoretical_cdf_vals)
        D_manual = np.max(discrepancies)
        
        # Compute p-value using one-sample Kolmogorov distribution
        # p-value = P(D > D_observed) for one-sample test
        p_value = ksone.sf(D_manual, n)  # sf = survival function = 1 - CDF
        
        print(f"\nKS Test (Type '{node_type}', k∈[{min_degree},{max_k}], {percent_used:.1f}% data):")
        print(f"  D statistic (manual): {D_manual:.6f}")
        print(f"  p-value: {p_value:.6f}")
        
        return D_manual, p_value

    def plot_ks_test(self, node_type='b', min_degree=0, min_bin_count=10, n_discrepancies=3, figsize=(12, 6)):
        """Plot empirical vs theoretical CDF with largest discrepancies highlighted."""
        from scipy.stats import kstest
        import matplotlib.patches as mpatches
        
        degrees = np.array(self._get_degrees(node_type))
        max_degree = int(np.max(degrees))
        
        # Find valid range
        k_values = np.arange(min_degree, max_degree + 1)
        observed_counts = np.array([np.sum(degrees == k) for k in k_values])
        valid_mask = observed_counts >= min_bin_count
        
        if not np.any(valid_mask):
            print(f"No bins with >= {min_bin_count} observations!")
            return None
        
        max_k = k_values[np.where(valid_mask)[0][-1]]
        
        # Filter to test range
        degrees_filtered = degrees[(degrees >= min_degree) & (degrees <= max_k)]
        percent_used = 100 * len(degrees_filtered) / len(degrees)
        
        # Compute CDFs at each unique degree value
        unique_degrees = np.sort(np.unique(degrees_filtered))
        empirical_cdf = np.array([np.sum(degrees_filtered <= k) / len(degrees_filtered) 
                                for k in unique_degrees])
        theoretical_cdf_vals = np.array([self.theoretical_cdf(k, node_type) 
                                        for k in unique_degrees])
        
        # Calculate discrepancies
        discrepancies = np.abs(empirical_cdf - theoretical_cdf_vals)
        
        # Find n largest discrepancies
        largest_indices = np.argsort(discrepancies)[-n_discrepancies:][::-1]
        
        # KS test
        result = kstest(degrees_filtered, 
                        lambda x: self.theoretical_cdf(x, node_type),
                        alternative='two-sided')
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot CDFs
        ax.step(unique_degrees, empirical_cdf, where='post', linewidth=2, 
                label='Empirical CDF', color='blue', alpha=0.7)
        ax.plot(unique_degrees, theoretical_cdf_vals, linewidth=2, 
                label='Theoretical CDF', color='red', alpha=0.7)
        
        # Colors for discrepancies
        colors = ['darkgreen', 'darkorange', 'purple', 'brown', 'pink']
        
        # Plot discrepancies with double-headed arrows
        legend_patches = []
        for i, idx in enumerate(largest_indices):
            k = unique_degrees[idx]
            emp = empirical_cdf[idx]
            theo = theoretical_cdf_vals[idx]
            disc = discrepancies[idx]
            color = colors[i % len(colors)]
            
            # Vertical line with arrows at both ends
            ax.annotate('', xy=(k, emp), xytext=(k, theo),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=2))
            
            # Create legend entry
            legend_patches.append(mpatches.Patch(color=color, label=f'k={int(k)}: D={disc:.4f}'))
        
        # Combined legend
        handles1, labels1 = ax.get_legend_handles_labels()
        all_handles = handles1 + legend_patches
        ax.legend(handles=all_handles, fontsize=10, loc='lower right', framealpha=0.9)
        
        ax.set_xlabel('In-degree k', fontsize=13)
        ax.set_ylabel('Cumulative Probability', fontsize=13)
        ax.set_title(f'KS Test: Type "{node_type}" (D={result.statistic:.4f}, p={result.pvalue:.4f}, '
                    f'{percent_used:.1f}% data)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=min_degree)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        print(f"\nPlot: Top {n_discrepancies} largest discrepancies:")
        for idx in largest_indices:
            k = unique_degrees[idx]
            disc = discrepancies[idx]
            print(f"  k={int(k):3d}: D = {disc:.4f}")
        
        return fig
    
    def monte_carlo_test(self, n_runs=10, min_degree=0, min_bin_count=10):
        """Chi-squared test stopping at first bin with < min_bin_count observations."""
        
        chi2_values = []
        p_values = []
        
        print(f"\nRunning {n_runs} Monte Carlo simulations...")
        
        for i in range(n_runs):
            # Generate new network
            net_mc = DirectedHomophilicNetwork(
                self.n0, self.n_nodes, self.m_edges, self.h, 
                self.f_a, self.mu['a'], self.mu['b']
            )
            net_mc.generate_network()
            
            # Get ALL type 'b' nodes (for proper scaling)
            all_degrees_b = np.array(net_mc._get_degrees('b'))
            n_total_b = len(all_degrees_b)
            
            # Find valid range
            max_degree = int(np.max(all_degrees_b))
            k_values = np.arange(min_degree, max_degree + 1)
            observed_full = np.array([np.sum(all_degrees_b == k) for k in k_values])
            
            # Find cutoff where bins drop below min_bin_count
            valid_mask = observed_full >= min_bin_count
            if not np.any(valid_mask):
                continue
            max_k_used = k_values[np.where(valid_mask)[0][-1]]
            
            # Filter to test range
            k_test = k_values[k_values <= max_k_used]
            observed_test = observed_full[k_values <= max_k_used]
            
            # Count how many nodes fall in test range
            n_in_test_range = np.sum((all_degrees_b >= min_degree) & (all_degrees_b <= max_k_used))
            
            # Expected: scale by nodes actually IN the test range
            theo_probs = np.array([self.theoretical_distribution(k, 'b') for k in k_test])
            theo_probs_normalized = theo_probs / theo_probs.sum()  # Conditional distribution
            expected_test = theo_probs_normalized * n_in_test_range
            
            # Chi-squared test
            chi2, p_value = chisquare(observed_test, expected_test)
            chi2_values.append(chi2)
            p_values.append(p_value)
        
        # Summary
        test_degrees = np.array(self._get_degrees('b'))
        total_b = len(test_degrees)
        
        n_in_range = np.sum((test_degrees >= min_degree) & (test_degrees <= max_k_used))
        percent_used = 100 * n_in_range / total_b if total_b > 0 else 0
        
        chi2_values = np.array(chi2_values)
        p_values = np.array(p_values)
        
        print(f"\nMonte Carlo ({len(chi2_values)} valid runs, k∈[{min_degree},{max_k_used}], {percent_used:.1f}% data):")
        print(f"  χ²: {np.mean(chi2_values):.2f} ± {np.std(chi2_values):.2f}")
        print(f"  p-value: {np.mean(p_values):.3f} ± {np.std(p_values):.3f}")
        print(f"  Fraction p > 0.05: {np.sum(p_values > 0.05) / len(chi2_values):.2%}\n")
        
        return chi2_values, p_values

    def print_statistics(self):
        """Print comprehensive network statistics."""
        in_degrees_a, in_degrees_b = self._get_degrees('a'), self._get_degrees('b')
        
        # Z-factor analysis
        if self.Z_factor:
            type_val = self.node_types[:self.graph.number_of_nodes()]
            in_deg = np.array([self.graph.in_degree(n) for n in self.graph.nodes()])
            
            lambda_vals = np.where(type_val == 0, self.lambda_a, self.lambda_b)
            mu_vals = np.where(type_val == 0, self.mu['a'], self.mu['b'])
            
            Z_emp = np.sum(lambda_vals * in_deg + mu_vals)
            Z_theo = self.graph.number_of_nodes() * self.Z_factor
            ratio = Z_emp / Z_theo
            
            print(f"\nZ-factor Analysis:")
            print(f"  Z_empirical/Z_theoretical = {ratio:.6f}")
            print(f"  % difference = {(ratio - 1.0) * 100:+.4f}%")
        
        # Type-specific statistics
        for degrees, type_name in [(in_degrees_a, 'a'), (in_degrees_b, 'b')]:
            n_total = self.graph.number_of_nodes()
            print(f"\nType '{type_name}': {len(degrees):,} nodes ({len(degrees)/n_total*100:.1f}%)")
            print(f"  Mean in-degree: {np.mean(degrees):.2f}")
            print(f"  Max in-degree: {max(degrees) if degrees else 0}")
            print(f"  Min in-degree: {min(degrees) if degrees else 'N/A'}")
        
        print(f"\ng_a = {self.g_a:.6f}, g_b (asymptotic) = {self.g_b_asymptotic:.6f}, g_a + g_b = {self.g_a + self.g_b_asymptotic:.6f}")

if __name__ == "__main__":
    # Generate network
    net = DirectedHomophilicNetwork(n0=100, n_nodes=10000, m_edges= 40, h=0.8, f_a=0.4, mu_a=1, mu_b=2)
    
    start = time.time()
    net.generate_network()


    degrees_b = np.array(net._get_degrees('b'))
    empirical_cdf_at_0 = np.sum(degrees_b <= 0) / len(degrees_b)
    theoretical_cdf_at_0 = net.theoretical_cdf(0, 'b')

    net.print_statistics()
    
    run_monte_carlo = False
    if run_monte_carlo:
        chi2_vals, p_vals = net.monte_carlo_test(n_runs=50, min_degree=0, min_bin_count=20)

    net.plot_ks_test(node_type='b', min_degree=0, min_bin_count= 0, n_discrepancies=3)
    net.ks_test(node_type='b', min_degree=0, min_bin_count=0)
    

    plt.show()
    

    net.plot_degree_distributions()
    plt.show()
    
    plot_Asymptotes = False
    if plot_Asymptotes:
        net.plot_in_edge_asymptotes()
        plt.show()
    
    plot_A_const = False
    if plot_A_const:
        net.plot_A_values()
        plt.show()