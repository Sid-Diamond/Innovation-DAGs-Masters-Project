import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from scipy.special import loggamma

class DirectedHomophilicNetwork:
    """
    Directed network with homophilic preferential attachment.
    Based on Price model structure with homophily parameters.
    """
    
    def __init__(self, n0, n_nodes, m_edges, h, f_a, mu_a, mu_b):
        """
        n0: initial number of nodes
        n_nodes: number of nodes to add
        m_edges: number of edges each new node creates (outgoing)
        h: homophily parameter (0.5= no preference, 1=complete homophily)
        f_a: fraction of "a" nodes
        mu_a, mu_b: fitness parameters for types a and b
        """
        self.n0 = n0
        self.n_nodes = n_nodes
        self.m_edges = m_edges
        self.h = h
        self.f_a = f_a
        self.f_b = 1 - f_a
        self.mu_a = mu_a
        self.mu_b = mu_b
        
        self.graph = None
        self.node_types = {}
        
        self.lambda_a = h * f_a + (1 - f_a) * (1 - h)
        self.lambda_b = h * (1 - f_a) + (1 - h) * f_a
        self.Z_factor = None
        self.Z_tilde = None
    
    def _compute_theoretical_params(self, g_a, g_b):
        """Compute Z̃ and theoretical exponents using asymptotic g_a, g_b."""
        m = self.m_edges

        self.g_a = g_a
        self.g_b = g_b
        
        # Z_factor = g_a*λ_a + g_b*λ_b + f_a*μ_a + f_b*μ_b
        term1 = g_a * self.lambda_a + g_b * self.lambda_b
        term2 = self.f_a * self.mu_a + self.f_b * self.mu_b
        self.Z_factor = term1 + term2
        
        # Z̃ = m/Z_factor
        self.Z_tilde = m / self.Z_factor
          
    def assign_node_type(self):
        """Randomly assign node type based on f_a."""
        return 'a' if np.random.rand() < self.f_a else 'b'
    
    def fit_asymptote(self, values, fraction=0.05):
        """Fit asymptote using mean of top fraction of values."""
        arr = np.array(values)
        n_tail = max(1, int(len(arr) * fraction))
        tail = arr[-n_tail:]
        return tail.mean()
    
    def homophilic_preferential_attachment(self, new_node_type, existing_nodes, t):
        """
        Select m_edges target nodes using homophilic preferential attachment.
        Π(target_i) = (λᵢ × kᵢ + μᵢ) / Z(t)
        """
        
        in_degrees = np.array([self.graph.in_degree(n) for n in existing_nodes])
        node_types = np.array([self.node_types[n] for n in existing_nodes])
        
        lambda_values = np.where(node_types == 'a', self.lambda_a, self.lambda_b)
        mu_values = np.where(node_types == 'a', self.mu_a, self.mu_b)
        
        numerators = lambda_values * in_degrees + mu_values
        
        # Use empirical Z for exact normalization
        Z_empirical = np.sum(numerators)
        probs = numerators / Z_empirical
        
        targets = np.random.choice(existing_nodes, self.m_edges, p=probs, replace=False)
        return targets

    def generate_network(self):
        """Generate directed network with homophilic preferential attachment."""
        self.graph = nx.DiGraph()
        self.edge_evolution = [] 

        # Initialize with n0 nodes and assign types based on f_a
        for i in range(self.n0):
            self.graph.add_node(i)
            self.node_types[i] = self.assign_node_type()

        # Add initial edges randomly among initial nodes
        for source in range(self.n0):
            targets = np.random.choice([t for t in range(self.n0) if t != source], size=self.m_edges, replace=False)
            for t in targets:
                self.graph.add_edge(source, t)

        # Add new nodes with homophilic preferential attachment
        for new_node in range(self.n0, self.n0 + self.n_nodes):
            t = new_node
            new_type = self.assign_node_type()
            self.node_types[new_node] = new_type
            
            existing_nodes = list(self.graph.nodes())
            self.graph.add_node(new_node)
            
            targets = self.homophilic_preferential_attachment(new_type, existing_nodes, t)
            
            for target in targets:
                self.graph.add_edge(new_node, target)
            
            # Track evolution every 100 nodes (or adjust frequency)
            if (new_node - self.n0) % 100 == 0 or new_node == self.n0 + self.n_nodes - 1:
                in_edges_a = sum(1 for _, target in self.graph.edges() if self.node_types[target] == 'a')
                in_edges_b = sum(1 for _, target in self.graph.edges() if self.node_types[target] == 'b')
                self.edge_evolution.append({'t': new_node, 'in_edges_a': in_edges_a, 'in_edges_b': in_edges_b})
        
        # After network generation, compute asymptotic values and then theoretical params
        mean_deg_a = np.array([d['in_edges_a']/d['t'] for d in self.edge_evolution])
        mean_deg_b = np.array([d['in_edges_b']/d['t'] for d in self.edge_evolution])
        
        g_a = self.fit_asymptote(mean_deg_a)
        g_b = self.fit_asymptote(mean_deg_b)
        
        self._compute_theoretical_params(g_a, g_b)

    def theoretical_distribution_a(self, k):
        """
        Theoretical in-degree distribution for "a" nodes.
        p(k) = A_a × Γ(k + α_a + γ_a) / Γ(k + α_a)
        """
        alpha_a = self.mu_a / self.lambda_a
        gamma_a = 1 + (1 /(self.Z_tilde*self.lambda_a))
        
        A_a_piece1 = 1 / (1 + (self.mu_a*self.Z_tilde))
        A_a_piece2 = 1 #(alpha_a)/(alpha_a + gamma_a)
        A_a_piece3 = gamma_func(alpha_a + gamma_a)/gamma_func(alpha_a)
        A_a = A_a_piece1 * A_a_piece2 * A_a_piece3
        
        log_ratio = loggamma(k + alpha_a) - loggamma(k + alpha_a + gamma_a)

        return A_a * np.exp(log_ratio)
    
    def theoretical_distribution_b(self, k):
        """
        Theoretical in-degree distribution for "b" nodes.
        p(k) = A_b × Γ(k + α_b + γ_b) / Γ(k + α_b)
        """
        alpha_b = self.mu_b / self.lambda_b
        gamma_b = 1 + (1 /(self.Z_tilde* self.lambda_b))

        A_b_piece1 = 1 / ((1 + (self.mu_b*self.Z_tilde)))
        A_b_piece2 = 1 #(alpha_b)/(alpha_b + gamma_b)
        A_b_piece3 = gamma_func(alpha_b + gamma_b)/gamma_func(alpha_b)
        A_b = A_b_piece1 * A_b_piece2 * A_b_piece3
        
        log_ratio = loggamma(k + alpha_b) - loggamma(k + alpha_b + gamma_b)
        
        return A_b * np.exp(log_ratio)
    
    def logarithmic_binning(self, degrees, bin_factor=1.1):
        """Create logarithmic bins for degree distribution."""
        degrees = np.array(degrees)
        if len(degrees) == 0:
            return np.array([]), np.array([])
        
        max_degree = np.max(degrees)
        n_total = len(degrees)
        
        bins = [0]
        current = 1
        while current <= max_degree:
            bins.append(int(current))
            current = current * bin_factor
        bins.append(int(max_degree) + 1)
        bins = sorted(list(set(bins)))
        
        bin_centers = []
        probabilities = []
        
        for i in range(len(bins) - 1):
            kmin = bins[i]
            kmax = bins[i + 1] - 1
            
            in_bin = (degrees >= kmin) & (degrees <= kmax)
            count = np.sum(in_bin)
            
            if count > 0:
                if kmin == 0:
                    center = np.sqrt(0.5 * kmax) if kmax > 0 else 0.5
                else:
                    center = np.sqrt(kmin * kmax)
                
                prob = count / n_total
                bin_centers.append(center)
                probabilities.append(prob)
        
        return np.array(bin_centers), np.array(probabilities)
    
    def plot_degree_distributions(self, figsize=(15, 6)):
        """Plot in-degree distributions for both node types with theoretical curves."""
        in_degrees_a = [self.graph.in_degree(n) for n in self.graph.nodes() 
                        if self.node_types[n] == 'a']
        in_degrees_b = [self.graph.in_degree(n) for n in self.graph.nodes() 
                        if self.node_types[n] == 'b']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot for type "a"
        if len(in_degrees_a) > 0:
            ax = axes[0]
            bin_centers_a, probs_a = self.logarithmic_binning(in_degrees_a)
            
            k_min_a = max(0, int(np.min(in_degrees_a)))
            k_max_a = int(np.max(in_degrees_a))
            k_range_a = np.arange(k_min_a, k_max_a + 1)
            
            theo_probs_a = np.array([self.theoretical_distribution_a(k) for k in k_range_a])
            
            ax.scatter(bin_centers_a, probs_a, s=50, alpha=0.7,
                    color='red', edgecolors='black', linewidths=0.5,
                    label='Simulation', zorder=3)
            
            mask = theo_probs_a > 0
            ax.plot(k_range_a[mask], theo_probs_a[mask],
                    '-', linewidth=2.5, color='darkred', alpha=0.85,
                    label='Theory', zorder=2)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=13)
            ax.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=13)
            ax.set_title(f'Type "a" (n={len(in_degrees_a)})', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        
        # Plot for type "b"
        if len(in_degrees_b) > 0:
            ax = axes[1]
            bin_centers_b, probs_b = self.logarithmic_binning(in_degrees_b)
            
            k_min_b = max(0, int(np.min(in_degrees_b)))
            k_max_b = int(np.max(in_degrees_b))
            k_range_b = np.arange(k_min_b, k_max_b + 1)
            
            theo_probs_b = np.array([self.theoretical_distribution_b(k) for k in k_range_b])
            
            ax.scatter(bin_centers_b, probs_b, s=50, alpha=0.7,
                    color='blue', edgecolors='black', linewidths=0.5,
                    label='Simulation', zorder=3)
            
            mask = theo_probs_b > 0
            ax.plot(k_range_b[mask], theo_probs_b[mask],
                    '-', linewidth=2.5, color='darkblue', alpha=0.85,
                    label='Theory', zorder=2)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=13)
            ax.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=13)
            ax.set_title(f'Type "b" (n={len(in_degrees_b)})', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        
        total_nodes = self.n0 + self.n_nodes
        fig.suptitle(
            f'Directed Homophilic Network: N={total_nodes:,}, m={self.m_edges}, '
            f'h={self.h}, f_a={self.f_a}',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        return fig

    def plot_in_edge_asymptotes(self, figsize=(10, 6), fraction=0.05):
        """
        Plot mean in-edge density for types a and b with asymptotic fits.
        """
        times = np.array([d['t'] for d in self.edge_evolution])
        mean_deg_a = np.array([d['in_edges_a']/d['t'] for d in self.edge_evolution])
        mean_deg_b = np.array([d['in_edges_b']/d['t'] for d in self.edge_evolution])

        # Compute asymptotes
        A_inf = self.fit_asymptote(mean_deg_a, fraction=fraction)
        B_inf = self.fit_asymptote(mean_deg_b, fraction=fraction)
        fig, ax = plt.subplots(figsize=figsize)

        # Plot data
        ax.plot(times, mean_deg_a, label=f"Type 'a' (data), m={self.m_edges}", color='red')
        ax.plot(times, mean_deg_b, label=f"Type 'b' (data), m={self.m_edges}", color='blue')

        # Overlay asymptotes
        ax.axhline(A_inf, linestyle='--', alpha=0.7, color='darkred',
                    label=f"Type 'a' asymptote = {A_inf:.3f}")
        ax.axhline(B_inf, linestyle='--', alpha=0.7, color='darkblue',
                    label=f"Type 'b' asymptote = {B_inf:.3f}")

        ax.set_xlabel("t (number of nodes)")
        ax.set_ylabel("Mean in-degree")
        ax.set_title("Asymptotic In-Edge Density")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def compute_A_values(self, max_k=50):
        """Compute A(k) values for both node types."""
        if self.Z_tilde is None:
            print("Error: Network must be generated first")
            return None
        
        k_values = np.arange(0, max_k + 1)
        results = {}
        
        for node_type in ['a', 'b']:
            # Get parameters
            if node_type == 'a':
                alpha = self.mu_a / self.lambda_a
                gamma_param = 1 + 1 / (self.Z_tilde * self.lambda_a)
                b0 = (self.m_edges * self.f_a) / (1 + (self.mu_a * self.Z_tilde))
            else:
                alpha = self.mu_b / self.lambda_b
                gamma_param = 1 + 1 / (self.Z_tilde * self.lambda_b)
                b0 = (self.m_edges * self.f_b) / (1 + (self.mu_b * self.Z_tilde))
            
            # Compute A(k)
            A_values = []
            for k in k_values:
                product = 1
                for i in range(k):
                    product *= (alpha + i) / (alpha + gamma_param + i)
                gamma_ratio = gamma_func(k + alpha + gamma_param) / gamma_func(k + alpha)
                A_values.append(b0 * product * gamma_ratio)
            
            A_theoretical = b0 * gamma_func(alpha + gamma_param) / gamma_func(alpha)
            b0_adjusted = b0 * gamma_func(alpha + gamma_param) / gamma_func(alpha)
            
            results[node_type] = {
                'A_values': np.array(A_values),
                'A_theoretical': A_theoretical,
                'b0_adjusted': b0_adjusted
            }
        
        return k_values, results

    def plot_A_values(self, max_k=50):
        """Plot A(k) for both node types."""
        k_values, results = self.compute_A_values(max_k)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, node_type in enumerate(['a', 'b']):
            ax = axes[idx]
            res = results[node_type]
            color = 'red' if node_type == 'a' else 'blue'
            
            ax.plot(k_values, res['A_values'], 'o-', color=color, 
                    linewidth=2, markersize=4, alpha=0.7, label='A(k) computed')
            ax.axhline(res['b0_adjusted'], linestyle='--', color='black', 
                    linewidth=2, alpha=0.7, 
                    label=f"b₀·Γ(α+γ)/Γ(α) = {res['b0_adjusted']:.2f}")
            
            ax.set_xlabel('k', fontsize=13)
            ax.set_ylabel('A(k)', fontsize=13)
            ax.set_title(f"Type '{node_type}' Normalization", fontsize=13, fontweight='bold')
            ax.set_xscale('log')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def print_statistics(self):
        """Print network statistics."""
        in_degrees_a = [self.graph.in_degree(n) for n in self.graph.nodes() 
                    if self.node_types[n] == 'a']
        in_degrees_b = [self.graph.in_degree(n) for n in self.graph.nodes() 
                    if self.node_types[n] == 'b']
        
        # Compute Z-factor check now
        if self.Z_factor is not None:
            in_degrees = np.array([self.graph.in_degree(n) for n in self.graph.nodes()])
            node_types = np.array([self.node_types[n] for n in self.graph.nodes()])
            
            lambda_values = np.where(node_types == 'a', self.lambda_a, self.lambda_b)
            mu_values = np.where(node_types == 'a', self.mu_a, self.mu_b)
            
            Z_empirical = np.sum(lambda_values * in_degrees + mu_values)
            Z_theoretical = self.graph.number_of_nodes() * self.Z_factor
            ratio = Z_empirical / Z_theoretical
            
            print(f"\nZ-factor Analysis:")
            print(f"  Z_empirical/Z_theoretical = {ratio:.6f}")
            print(f"  % difference = {(ratio - 1.0) * 100:+.4f}%")

        print(f"\nType 'a': {len(in_degrees_a):,} nodes ({len(in_degrees_a)/self.graph.number_of_nodes()*100:.1f}%)")
        print(f"  Mean in-degree: {np.mean(in_degrees_a):.2f}")
        print(f"  Max in-degree: {max(in_degrees_a) if in_degrees_a else 0}")
        print(f"  Min in-degree type 'a': {min(in_degrees_a) if in_degrees_a else 'N/A'}")

        
        print(f"\nType 'b': {len(in_degrees_b):,} nodes ({len(in_degrees_b)/self.graph.number_of_nodes()*100:.1f}%)")
        print(f"  Mean in-degree: {np.mean(in_degrees_b):.2f}")
        print(f"  Max in-degree: {max(in_degrees_b) if in_degrees_b else 0}")
        print(f"  Min in-degree type 'b': {min(in_degrees_b) if in_degrees_b else 'N/A'}")

        print(f"\ng_a = {self.g_a:.6f}, g_b = {self.g_b:.6f}, g_a + g_b = {self.g_a + self.g_b:.6f}")

if __name__ == "__main__":
    # Generate network with homophily
    net = DirectedHomophilicNetwork(
        n0=40, 
        n_nodes=5000, 
        m_edges=30, 
        h=0.8,
        f_a=0.6,
        mu_a=2,
        mu_b=1
    )
    
    net.generate_network()
    net.print_statistics()
     
    # Plot
    fig = net.plot_degree_distributions()
    plt.show()
    fig = net.plot_in_edge_asymptotes()
    plt.show()
    fig = net.plot_A_values(max_k=50)
    plt.show()