import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func, beta as beta_func
from scipy.stats import chisquare, ksone
from typing import Dict, Tuple, List
import time

from sklearn.linear_model import Log

class DirectedHomophilicNetwork:
    """Optimized directed network with homophilic preferential attachment."""
    
    def __init__(self, n0: int, n_nodes: int, m_edges: int, h: float, f_a: float, mu_a: float, mu_b: float, seed: int = None):
        # Network parameters
        self.n0, self.n_nodes, self.m_edges = n0, n_nodes, m_edges
        self.h, self.f_a, self.f_b = h, f_a, 1 - f_a
        self.mu = {'a': mu_a, 'b': mu_b}
        self.seed = seed
        
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Computed parameters
        self.lambda_a = h * f_a + (1 - f_a) * (1 - h)
        self.lambda_b = h * self.f_b + (1- self.f_b)*(1 - h)
        self.lambda_ = {'a': self.lambda_a, 'b': self.lambda_b}
        
        # Network state
        self.graph = None
        self.node_types = None
        self.edge_evolution = []
        self.g_a, self.g_b, self.g_b_empirical, self.Z_factor, self.Z_tilde = None, None, None, None, None
                
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
        p0 = 1 / (1 + mu_x * self.Z_tilde)
        A = p0 * gamma_func(alpha + gamma) / gamma_func(alpha)
        
        return {'alpha': alpha, 'gamma': gamma, 'p0': p0, 'A': A}

    def _compute_theoretical_params(self, g_a: float, g_b: float, g_b_empirical: float):
        """Compute Z̃ using asymptotic g values."""
        self.g_a = g_a
        self.g_b = g_b
        self.g_b_empirical = g_b_empirical  # Store for comparison only
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
        self.node_types = np.empty(total_nodes, dtype=np.int8) # 0 for 'a', 1 for 'b'
        self.in_degrees = np.zeros(total_nodes, dtype=np.int32)
        self.graph = nx.DiGraph()
        
        # Initialize nodes
        for i in range(self.n0):
            self.node_types[i] = 0 if self.assign_node_type() == 'a' else 1
            self.graph.add_node(i)
        
        # Initial random edges
        for source in range(self.n0):
            targets = np.random.choice([t for t in range(self.n0) if t != source], size=self.m_edges, replace=False)
            # select m targets without self-loops without seleccting same target multiple times. Node type irrelevant here.
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
                # Stores in edge counts for every y = n0 +100t and at the final datapoint for a non memory 
                # intensive asymptotic g value calculation
        
        self._fit_asymptotes()
    
    def _fit_asymptotes(self, fraction: float = 0.05):
        """Fit asymptotic g values from evolution data."""
        mean_deg_a = np.array([d['in_edges_a']/d['t'] for d in self.edge_evolution])
        mean_deg_b = np.array([d['in_edges_b']/d['t'] for d in self.edge_evolution])
        
        n_tail = max(1, int(len(mean_deg_a) * fraction))
        g_a = mean_deg_a[-n_tail:].mean()
        g_b_empirical = mean_deg_b[-n_tail:].mean()  # Store for comparison
        g_b = self.m_edges - g_a  # Enforce constraint
        
        self._compute_theoretical_params(g_a, g_b, g_b_empirical)

    def theoretical_distribution(self, k, node_type: str):
        """
        Theoretical in-degree distribution with analytic continuation.
        Form: p(k) = p_0 * B(k, alpha + gamma) / B(k, alpha) for k > 0. p0 at k=0.
        """
        params = self._get_params(node_type)
        k = np.atleast_1d(k)
        result = np.zeros_like(k, dtype=float)
        
        zero_mask = (k == 0)
        pos_mask = (k > 0)
        
        result[zero_mask] = params['p0']
        if np.any(pos_mask):
            k_pos = k[pos_mask]
            result[pos_mask] = params['p0'] * beta_func(k_pos, params['alpha'] + params['gamma']) / beta_func(k_pos, params['alpha'])
        
        return result.item() if result.shape == (1,) else result

    def theoretical_cdf_discrete(self, k, node_type: str, kmin: int, kmax: int):
            """
            Discrete CDF for the Yule–Simon–type distribution. Defines the conditional CDF:
            F(k) = P(K <= k | kmin <= K <= kmax) where the PMF is renormalised over [kmin, kmax].
            """

            # Ensure array-like input
            k = np.atleast_1d(k)
            k_int = np.floor(k).astype(int)

            # Discrete support
            support = np.arange(kmin, kmax + 1)

            # Evaluate PMF on integer support
            pmf = np.array([self.theoretical_distribution(j, node_type) for j in support], dtype=float)

            # Renormalisation over truncated support
            Z = pmf.sum()
            pmf /= Z
            cdf = np.cumsum(pmf)

            # Map k -> CDF value
            F = np.zeros_like(k_int, dtype=float)
            for i, ki in enumerate(k_int):
                if ki < kmin:
                    F[i] = 0.0
                elif ki >= kmax:
                    F[i] = 1.0
                else:
                    F[i] = cdf[ki - kmin]

            return F.item() if F.size == 1 else F

    def empirical_cdf_integers(self, data):
        """
        Compute the empirical CDF for integer data only (vectorized).
        """
        data = np.array(data, dtype=int)
        n = len(data)
        if n == 0:
            return np.array([]), np.array([])

        unique_vals = np.arange(data.min(), data.max() + 1)
        # Vectorized: use searchsorted to count how many elements <= each unique value
        sorted_data = np.sort(data)
        counts = np.searchsorted(sorted_data, unique_vals, side='right')
        cdf_vals = counts / n

        return unique_vals, cdf_vals

    def _get_degrees(self, node_type: str) -> List[int]:
        """Get in-degrees for nodes of specified type."""
        type_val = 0 if node_type == 'a' else 1
        return [self.graph.in_degree(n) for n in self.graph.nodes() 
                if self.node_types[n] == type_val]
    
    def logarithmic_binning(self, degrees: np.ndarray, bin_factor: float = 1.01) -> Tuple[np.ndarray, np.ndarray]:
        """Create logarithmic bins with k=0 always in its own bin."""
        if len(degrees) == 0:
            return np.array([]), np.array([])# prevents crash out if no nodes of a given type exist
        
        degrees = np.array(degrees)
        max_degree, n_total = np.max(degrees), len(degrees)
        
        # Handle k=0 separately
        count_zero = np.sum(degrees == 0)
        bin_centers = [0.0] if count_zero > 0 else []
        probabilities = [count_zero / n_total] if count_zero > 0 else []
        
        # Create logarithmic sequence for k >= 1
        if np.any(degrees > 0):
            bins = [1]
            current = 1 * bin_factor
            while current <= max_degree:
                bins.append(int(current))
                current *= bin_factor
            bins.append(int(max_degree) + 1)
            bins = sorted(set(bins))
            
            # Compute bin statistics for k > 0: turn the logarithmic sequence into bins 
            # and count how many degrees fall into each bin
            for i in range(len(bins) - 1):
                kmin, kmax = bins[i], bins[i + 1] - 1
                count = np.sum((degrees >= kmin) & (degrees <= kmax))
                
                if count > 0:
                    center = np.sqrt(kmin * kmax)  # Geometric mean (always kmin > 0)
                    bin_centers.append(center)
                    probabilities.append(count / n_total)
        
        return np.array(bin_centers), np.array(probabilities)

    def ks_test_and_plot(self, node_type='b', kmin=0, kmax=25, figsize=(12, 6)):
        """
        KS test with visualization over integer degrees in [kmin, kmax].
        Only the KS test decides the range; theoretical CDF is evaluated at these points.
        """
        
        degrees = np.array(self._get_degrees(node_type), dtype=int)
        if len(degrees) == 0:
            print("No nodes of this type!")
            return None, None, None

        if kmax is None:
            kmax = int(degrees.max())

        # Filter degrees to the test range
        degrees_filtered = degrees[(degrees >= kmin) & (degrees <= kmax)]
        if len(degrees_filtered) == 0:
            print(f"No degrees in range [{kmin}, {kmax}]")
            return None, None, None

        n = len(degrees_filtered)
        percent_used = 100 * n / len(degrees)

        # Compute empirical CDF over filtered integers (vectorized)
        unique_degrees = np.arange(kmin, kmax + 1)
        sorted_data = np.sort(degrees_filtered)
        counts = np.searchsorted(sorted_data, unique_degrees, side='right')
        empirical_cdf = counts / n

        # Theoretical CDF values at unique degrees
        theoretical_cdf_vals = self.theoretical_cdf_discrete(unique_degrees, node_type, kmin=kmin, kmax=kmax)
        
        # KS statistic
        discrepancies = np.abs(empirical_cdf - theoretical_cdf_vals)
        D = discrepancies.max()
        p_value = ksone.sf(D, n)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(unique_degrees, empirical_cdf, 'o-', label='Empirical CDF', color='blue', alpha=0.7)
        ax.plot(unique_degrees, theoretical_cdf_vals, '-', label='Theoretical CDF', color='red', alpha=0.7)

        ax.set_xlabel('In-degree k')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'KS Test: Type "{node_type}" (D={D:.4f}, p={p_value:.2e})')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=kmin, right=kmax)
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()

        print(f"\nKS Test: Type '{node_type}', k∈[{kmin},{kmax}], {percent_used:.1f}% data")
        print(f"  D = {D:.6f}, p = {p_value:.6e}")

        return fig, D, p_value

    def plot_degree_distributions(self, figsize: Tuple = (15, 6), discretisations: int = 10**5):
        """
        Plot in-degree distributions with theoretical curves. If discretisations=0,
        theoretical curve is only computed at integer k.
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
                A_values.append(params['p0'] * product * gamma_ratio)
            
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
            theo_probs_normalized = theo_probs / theo_probs.sum()
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
        
        # g_b comparison
        if self.g_b_empirical is not None:
            ratio_gb = self.g_b / self.g_b_empirical
            print(f"\ng_b Analysis:")
            print(f"  g_b (m - g_a)/g_b_empirical = {ratio_gb:.6f}")
            print(f"  % difference = {(ratio_gb - 1.0) * 100:+.4f}%")
        
        # Type-specific statistics
        for degrees, type_name in [(in_degrees_a, 'a'), (in_degrees_b, 'b')]:
            n_total = self.graph.number_of_nodes()
            print(f"\nType '{type_name}': {len(degrees):,} nodes ({len(degrees)/n_total*100:.1f}%)")
            print(f"  Mean in-degree: {np.mean(degrees):.2f}")
            print(f"  Max in-degree: {max(degrees) if degrees else 0}")
            print(f"  Min in-degree: {min(degrees) if degrees else 'N/A'}")
        
        print(f"\ng_a = {self.g_a:.6f}, g_b = {self.g_b:.6f}, g_b_empirical = {self.g_b_empirical:.6f}")
        print(f"g_a + g_b = {self.g_a + self.g_b:.6f} (m = {self.m_edges})")

    def plot_degree_distributions_discrete(self, figsize: Tuple = (15, 6), max_k_display: int = None):
        """
        Plot using discrete integer probabilities - no binning.
        This shows the true empirical PMF at each integer k value.
        LINEAR SCALE VERSION - raw data.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
            in_degrees = self._get_degrees(node_type)
            if len(in_degrees) == 0:
                continue
                
            ax = axes[idx]
            
            # Empirical PMF at integer values
            unique_k, counts = np.unique(in_degrees, return_counts=True)
            empirical_pmf = counts / len(in_degrees)
            
            ax.scatter(unique_k, empirical_pmf, s=1, alpha=0.7, color=color,
                    edgecolors='black', linewidths=1, label='Simulation', zorder=3)
            
            # Theoretical curve at integer values
            k_max = int(np.max(in_degrees)) if max_k_display is None else max_k_display
            k_range = np.arange(0, k_max + 1)
            theo_probs = self.theoretical_distribution(k_range, node_type)
            
            ax.plot(k_range, theo_probs, '-', linewidth=1, color='dark'+color, 
                    alpha=1, label='Theory', zorder=2)
            
            # Formatting - LINEAR AXES
            ax.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=13)
            ax.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=13)
            ax.set_title(f'Type "{node_type}" (n={len(in_degrees)}) - LINEAR SCALE', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        fig.suptitle(f'Directed Homophilic Network: N={self.n0 + self.n_nodes:,}, '
                    f'm={self.m_edges}, h={self.h}, f_a={self.f_a}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_degree_distributions_hybrid(self, figsize: Tuple = (15, 12), max_k_display: int = None):
        """
        Plot using discrete integer probabilities - TWO VIEWS.
        Top row: log-log for full range
        Bottom row: linear scale zoomed to where you have good statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
            in_degrees = self._get_degrees(node_type)
            if len(in_degrees) == 0:
                continue
                
            # Empirical PMF at integer values
            unique_k, counts = np.unique(in_degrees, return_counts=True)
            empirical_pmf = counts / len(in_degrees)
            
            # Theoretical curve at integer values
            k_max = int(np.max(in_degrees)) if max_k_display is None else max_k_display
            k_range = np.arange(0, k_max + 1)
            theo_probs = self.theoretical_distribution(k_range, node_type)
            
            # TOP ROW: Log-log (full range)
            ax_log = axes[0, idx]
            ax_log.scatter(unique_k, empirical_pmf, s=50, alpha=0.7, color=color,
                        edgecolors='black', linewidths=0.5, label='Simulation', zorder=3)
            mask = theo_probs > 0
            ax_log.plot(k_range[mask], theo_probs[mask], '-', linewidth=2.5, color='dark'+color, 
                        alpha=0.85, label='Theory', zorder=2)
            ax_log.set_xscale('log')
            ax_log.set_yscale('log')
            ax_log.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=11)
            ax_log.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=11)
            ax_log.set_title(f'Type "{node_type}" - LOG SCALE (full range)', fontsize=11, fontweight='bold')
            ax_log.legend(fontsize=9)
            ax_log.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
            
            # BOTTOM ROW: Linear (zoomed to good statistics)
            ax_lin = axes[1, idx]
            # Only show k where you have at least 5 observations
            good_stats_mask = counts >= 5
            k_cutoff = unique_k[good_stats_mask][-1] if np.any(good_stats_mask) else 50
            k_cutoff = min(k_cutoff, 100)  # Cap at k=100 for readability
            
            # Plot empirical
            plot_mask = unique_k <= k_cutoff
            ax_lin.scatter(unique_k[plot_mask], empirical_pmf[plot_mask], s=50, alpha=0.7, color=color,
                        edgecolors='black', linewidths=0.5, label='Simulation', zorder=3)
            
            # Plot theory up to cutoff
            k_range_zoom = np.arange(0, k_cutoff + 1)
            theo_probs_zoom = self.theoretical_distribution(k_range_zoom, node_type)
            ax_lin.plot(k_range_zoom, theo_probs_zoom, '-', linewidth=2.5, color='dark'+color, 
                        alpha=0.85, label='Theory', zorder=2)
            
            ax_lin.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=11)
            ax_lin.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=11)
            ax_lin.set_title(f'Type "{node_type}" - LINEAR (k ≤ {k_cutoff}, good stats)', fontsize=11, fontweight='bold')
            ax_lin.legend(fontsize=9)
            ax_lin.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_lin.set_xlim(-1, k_cutoff + 5)
        
        fig.suptitle(f'Directed Homophilic Network: N={self.n0 + self.n_nodes:,}, '
                    f'm={self.m_edges}, h={self.h}, f_a={self.f_a}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def ks_sweep_m_edges(self, m_min, m_max, m_step=1, node_type='b', kmin=0, kmax=25, 
                        n_runs=3, figsize=(12, 6)):
        """
        Sweep over different m_edges values and compute KS statistics.
        dict : Results containing m values, mean/std of D and p-values
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        m_values = np.arange(m_min, m_max + 1, m_step)
        D_means, D_stds = [], []
        p_means, p_stds = [], []
        
        print(f"\nSweeping m_edges: {m_values}")
        print(f"Fixed: n0={self.n0}, n_nodes={self.n_nodes}, h={self.h}, f_a={self.f_a}")
        
        for m in m_values:
            D_runs, p_runs = [], []
            
            for run in range(n_runs):
                # Generate network with this m value
                net_temp = DirectedHomophilicNetwork(
                    self.n0, self.n_nodes, int(m), self.h, 
                    self.f_a, self.mu['a'], self.mu['b']
                )
                net_temp.generate_network()
                
                # Get degrees and run KS test
                degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                if len(degrees) == 0:
                    continue
                
                # Auto-set kmax if None
                kmax_actual = int(degrees.max()) if kmax is None else kmax
                
                degrees_filtered = degrees[(degrees >= kmin) & (degrees <= kmax_actual)]
                if len(degrees_filtered) == 0:
                    continue
                
                n = len(degrees_filtered)
                unique_degrees = np.arange(kmin, kmax_actual + 1)
                sorted_data = np.sort(degrees_filtered)
                counts = np.searchsorted(sorted_data, unique_degrees, side='right')
                empirical_cdf = counts / n
                
                theoretical_cdf_vals = net_temp.theoretical_cdf_discrete(
                    unique_degrees, node_type, kmin=kmin, kmax=kmax_actual
                )
                
                D = np.abs(empirical_cdf - theoretical_cdf_vals).max()
                p_value = ksone.sf(D, n)
                
                D_runs.append(D)
                p_runs.append(p_value)
            
            if len(D_runs) > 0:
                D_means.append(np.mean(D_runs))
                D_stds.append(np.std(D_runs))
                p_means.append(np.mean(p_runs))
                p_stds.append(np.std(p_runs))
                print(f"  m={m:2d}: D={D_means[-1]:.4f}±{D_stds[-1]:.4f}, p={p_means[-1]:.3f}±{p_stds[-1]:.3f}")
        
        # Convert to arrays
        D_means = np.array(D_means)
        D_stds = np.array(D_stds)
        p_means = np.array(p_means)
        p_stds = np.array(p_stds)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # KS statistic
        if n_runs == 1:
            ax1.plot(m_values, D_means, marker='o', linewidth=2, markersize=8, label='KS Statistic D')
        else:
            ax1.errorbar(m_values, D_means, yerr=D_stds, marker='o', capsize=5, 
                        linewidth=2, markersize=8, label='KS Statistic D')
        ax1.set_xlabel('m_edges', fontsize=12)
        ax1.set_ylabel('KS Statistic D', fontsize=12)
        ax1.set_title(f'KS Test vs m_edges (Type "{node_type}")', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # p-value
        if n_runs == 1:
            ax2.plot(m_values, p_means, marker='s', linewidth=2, markersize=8, color='red', label='p-value')
        else:
            ax2.errorbar(m_values, p_means, yerr=p_stds, marker='s', capsize=5,
                        linewidth=2, markersize=8, color='red', label='p-value')
        ax2.axhline(0.05, linestyle='--', color='black', alpha=0.5, label='p=0.05')
        ax2.set_xlabel('m_edges', fontsize=12)
        ax2.set_ylabel('p-value', fontsize=12)
        ax2.set_title(f'p-value vs m_edges (Type "{node_type}")', fontsize=13, fontweight='bold')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        return {
            'm_values': m_values,
            'D_means': D_means,
            'D_stds': D_stds,
            'p_means': p_means,
            'p_stds': p_stds,
            'fig': fig
        }

    def ks_sweep_n0(self, n0_min, n0_max, n0_step=1, node_type='b', kmin=0, kmax=25,
                    n_runs=3, figsize=(12, 6)):
        """
        Sweep over different n0 values and compute KS statistics.
        dict : Results containing n0 values, mean/std of D and p-values
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        n0_values = np.arange(n0_min, n0_max + 1, n0_step)
        D_means, D_stds = [], []
        p_means, p_stds = [], []
        
        print(f"\nSweeping n0: {n0_values}")
        print(f"Fixed: m_edges={self.m_edges}, n_nodes={self.n_nodes}, h={self.h}, f_a={self.f_a}")
        
        for n0 in n0_values:
            D_runs, p_runs = [], []
            
            for run in range(n_runs):
                # Generate network with this n0 value
                net_temp = DirectedHomophilicNetwork(
                    int(n0), self.n_nodes, self.m_edges, self.h,
                    self.f_a, self.mu['a'], self.mu['b']
                )
                net_temp.generate_network()
                
                # Get degrees and run KS test
                degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                if len(degrees) == 0:
                    continue
                
                # Auto-set kmax if None
                kmax_actual = int(degrees.max()) if kmax is None else kmax
                
                degrees_filtered = degrees[(degrees >= kmin) & (degrees <= kmax_actual)]
                if len(degrees_filtered) == 0:
                    continue
                
                n = len(degrees_filtered)
                unique_degrees = np.arange(kmin, kmax_actual + 1)
                sorted_data = np.sort(degrees_filtered)
                counts = np.searchsorted(sorted_data, unique_degrees, side='right')
                empirical_cdf = counts / n
                
                theoretical_cdf_vals = net_temp.theoretical_cdf_discrete(
                    unique_degrees, node_type, kmin=kmin, kmax=kmax_actual
                )
                
                D = np.abs(empirical_cdf - theoretical_cdf_vals).max()
                p_value = ksone.sf(D, n)
                
                D_runs.append(D)
                p_runs.append(p_value)
            
            if len(D_runs) > 0:
                D_means.append(np.mean(D_runs))
                D_stds.append(np.std(D_runs))
                p_means.append(np.mean(p_runs))
                p_stds.append(np.std(p_runs))
                print(f"  n0={n0:4d}: D={D_means[-1]:.4f}±{D_stds[-1]:.4f}, p={p_means[-1]:.3f}±{p_stds[-1]:.3f}")
        
        # Convert to arrays
        D_means = np.array(D_means)
        D_stds = np.array(D_stds)
        p_means = np.array(p_means)
        p_stds = np.array(p_stds)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # KS statistic
        if n_runs == 1:
            ax1.plot(n0_values, D_means, marker='o', linewidth=2, markersize=8, label='KS Statistic D')
        else:
            ax1.errorbar(n0_values, D_means, yerr=D_stds, marker='o', capsize=5,
                        linewidth=2, markersize=8, label='KS Statistic D')
        ax1.set_xlabel('n0 (initial nodes)', fontsize=12)
        ax1.set_ylabel('KS Statistic D', fontsize=12)
        ax1.set_title(f'KS Test vs n0 (Type "{node_type}")', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # p-value
        if n_runs == 1:
            ax2.plot(n0_values, p_means, marker='s', linewidth=2, markersize=8, color='red', label='p-value')
        else:
            ax2.errorbar(n0_values, p_means, yerr=p_stds, marker='s', capsize=5,
                        linewidth=2, markersize=8, color='red', label='p-value')
        ax2.axhline(0.05, linestyle='--', color='black', alpha=0.5, label='p=0.05')
        ax2.set_xlabel('n0 (initial nodes)', fontsize=12)
        ax2.set_ylabel('p-value', fontsize=12)
        ax2.set_title(f'p-value vs n0 (Type "{node_type}")', fontsize=13, fontweight='bold')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        return {
            'n0_values': n0_values,
            'D_means': D_means,
            'D_stds': D_stds,
            'p_means': p_means,
            'p_stds': p_stds,
            'fig': fig
        }

    def ks_surface_n0_m(self, n0_min, n0_max, n0_step, m_min, m_max, m_step, 
                        node_type='b', kmin=0, kmax=25, n_runs=3, figsize=(16, 6)):
        """
        dict : Results containing grid values and p-value/D statistic surfaces
        """
        import warnings
        warnings.filterwarnings('ignore')
        from mpl_toolkits.mplot3d import Axes3D
        
        n0_values = np.arange(n0_min, n0_max + 1, n0_step)
        m_values = np.arange(m_min, m_max + 1, m_step)
        
        # Create meshgrid for surface
        N0_grid, M_grid = np.meshgrid(n0_values, m_values, indexing='ij')
        P_grid = np.zeros_like(N0_grid, dtype=float)
        D_grid = np.zeros_like(N0_grid, dtype=float)
        
        print(f"\nCreating surface: n0={n0_values}, m={m_values}")
        print(f"Fixed: n_nodes={self.n_nodes}, h={self.h}, f_a={self.f_a}")
        print(f"Total points: {len(n0_values) * len(m_values)}\n")
        
        point_count = 0
        total_points = len(n0_values) * len(m_values)
        
        for i, n0 in enumerate(n0_values):
            for j, m in enumerate(m_values):
                D_runs, p_runs = [], []
                
                for run in range(n_runs):
                    # Generate network
                    net_temp = DirectedHomophilicNetwork(
                        int(n0), self.n_nodes, int(m), self.h,
                        self.f_a, self.mu['a'], self.mu['b']
                    )
                    net_temp.generate_network()
                    
                    # KS test
                    degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                    if len(degrees) == 0:
                        continue
                    
                    # Auto-set kmax if None
                    kmax_actual = int(degrees.max()) if kmax is None else kmax
                    
                    degrees_filtered = degrees[(degrees >= kmin) & (degrees <= kmax_actual)]
                    if len(degrees_filtered) == 0:
                        continue
                    
                    n = len(degrees_filtered)
                    unique_degrees = np.arange(kmin, kmax_actual + 1)
                    sorted_data = np.sort(degrees_filtered)
                    counts = np.searchsorted(sorted_data, unique_degrees, side='right')
                    empirical_cdf = counts / n
                    
                    theoretical_cdf_vals = net_temp.theoretical_cdf_discrete(
                        unique_degrees, node_type, kmin=kmin, kmax=kmax_actual
                    )
                    
                    D = np.abs(empirical_cdf - theoretical_cdf_vals).max()
                    p_value = ksone.sf(D, n)
                    
                    D_runs.append(D)
                    p_runs.append(p_value)
                
                if len(D_runs) > 0:
                    P_grid[i, j] = np.mean(p_runs)
                    D_grid[i, j] = np.mean(D_runs)
                else:
                    P_grid[i, j] = np.nan
                    D_grid[i, j] = np.nan
                
                point_count += 1
                if point_count % 5 == 0 or point_count == total_points:
                    print(f"  Progress: {point_count}/{total_points} ({100*point_count/total_points:.1f}%)")
        
        # Create visualizations
        fig = plt.figure(figsize=figsize)
        
        # 3D surface for p-value
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(N0_grid, M_grid, P_grid, cmap='viridis', 
                                edgecolor='none', alpha=0.8)
        ax1.set_xlabel('n0', fontsize=11)
        ax1.set_ylabel('m_edges', fontsize=11)
        ax1.set_zlabel('p-value', fontsize=11)
        ax1.set_title(f'KS p-value Surface (Type "{node_type}")', fontsize=12, fontweight='bold')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        
        # 3D surface for D statistic
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(N0_grid, M_grid, D_grid, cmap='plasma',
                                edgecolor='none', alpha=0.8)
        ax2.set_xlabel('n0', fontsize=11)
        ax2.set_ylabel('m_edges', fontsize=11)
        ax2.set_zlabel('KS Statistic D', fontsize=11)
        ax2.set_title(f'KS D Statistic Surface (Type "{node_type}")', fontsize=12, fontweight='bold')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        # Create contour plot as well
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
        
        # p-value contour
        contour1 = ax3.contourf(M_grid, N0_grid, P_grid, levels=20, cmap='viridis')
        ax3.contour(M_grid, N0_grid, P_grid, levels=[0.05], colors='red', 
                    linewidths=2, linestyles='--')
        ax3.set_xlabel('m_edges', fontsize=12)
        ax3.set_ylabel('n0', fontsize=12)
        ax3.set_title(f'p-value Contours (Type "{node_type}")\nRed line: p=0.05', 
                    fontsize=12, fontweight='bold')
        fig2.colorbar(contour1, ax=ax3)
        
        # D statistic contour
        contour2 = ax4.contourf(M_grid, N0_grid, D_grid, levels=20, cmap='plasma')
        ax4.set_xlabel('m_edges', fontsize=12)
        ax4.set_ylabel('n0', fontsize=12)
        ax4.set_title(f'KS D Statistic Contours (Type "{node_type}")', 
                    fontsize=12, fontweight='bold')
        fig2.colorbar(contour2, ax=ax4)
        
        plt.tight_layout()
        
        return {
            'n0_grid': N0_grid,
            'm_grid': M_grid,
            'p_grid': P_grid,
            'd_grid': D_grid,
            'fig_3d': fig,
            'fig_contour': fig2
        }

if __name__ == "__main__":
    net = DirectedHomophilicNetwork(n0=100, n_nodes=30000, m_edges=10, h=0.8, f_a=0.8, mu_a=5, mu_b=1, seed=5) 
    start = time.time()
    net.generate_network()
    print(f"Network generated in {time.time() - start:.2f}s")
    
    Statistics = True
    if Statistics:
        net.print_statistics()
    
    Log_Binned = False
    if Log_Binned:
        net.plot_degree_distributions()
        plt.show()

    Discrete_Linear = False
    if Discrete_Linear:
        net.plot_degree_distributions_discrete()
        plt.show()

    Hybrid_Plot = True
    if Hybrid_Plot:
        net.plot_degree_distributions_hybrid()
        plt.show()
    
    KS_test = True
    if KS_test:
        fig, D, p = net.ks_test_and_plot(node_type='b', kmin=0, kmax=None)
        plt.show()

    run_monte_carlo = False
    if run_monte_carlo:
        chi2_vals, p_vals = net.monte_carlo_test(n_runs=50, min_degree=0, min_bin_count=0)
    
    plot_asymptotes = False
    if plot_asymptotes:
        net.plot_in_edge_asymptotes()
        plt.show()
    
    plot_A_const = False
    if plot_A_const:
        net.plot_A_values()
        plt.show()
    

    sweep_m_edges = True
    if sweep_m_edges: 
        results_m = net.ks_sweep_m_edges(m_min=5, m_max=20, m_step=2, node_type='b', kmin=0, kmax= None, n_runs=5)
        plt.show()
    
    sweep_n0 = False
    if sweep_n0:
        results_n0 = net.ks_sweep_n0(n0_min=50, n0_max=500, n0_step=50, node_type='b', kmin=0, kmax=25, n_runs=3)
        plt.show()
    
    surface_n0_m = False
    if surface_n0_m:
        results_surface = net.ks_surface_n0_m(n0_min=50, n0_max=500, n0_step=100,m_min=5,m_max=20, m_step=5, node_type='b',kmin=0,kmax=None, n_runs=2)
        plt.show()