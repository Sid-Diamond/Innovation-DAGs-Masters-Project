import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func

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
        self.z_ratios = []
        
        self._compute_theoretical_params()
    
    def _compute_theoretical_params(self):
        """Compute λₐ, λᵦ, Z̃ and theoretical exponents."""
        f = self.f_a
        h = self.h
        m = self.m_edges
        
        self.lambda_a = h * f + (1 - f) * (1 - h)
        self.lambda_b = h * (1 - f) + (1 - h) * f
        
        # Z normalization factor
        term1 = m * (self.lambda_a**2 + self.lambda_b**2)
        term2 = f * self.mu_a + (1 - f) * self.mu_b
        self.Z_factor = term1 + term2
        
        # Z̃ = m/Z_factor
        self.Z_tilde = m/self.Z_factor
          
    def assign_node_type(self):
        """Randomly assign node type based on f_a."""
        return 'a' if np.random.rand() < self.f_a else 'b'
    
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
        
        # Track theoretical vs empirical Z
        Z_theoretical = t * self.Z_factor
        Z_empirical = np.sum(numerators)
        ratio = Z_empirical / Z_theoretical
        self.z_ratios.append(ratio)
        
        # Use empirical Z for exact normalization
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
        
        initial_E = self.graph.number_of_edges()
        print(f"Initial edges = {initial_E},   expected m*n0 = {self.m_edges * self.n0}")


        # Track initial state
        t = self.n0
        in_edges_a = sum(1 for _, target in self.graph.edges() if self.node_types[target] == 'a')
        in_edges_b = sum(1 for _, target in self.graph.edges() if self.node_types[target] == 'b')
        self.edge_evolution.append({'t': t, 'in_edges_a': in_edges_a, 'in_edges_b': in_edges_b})

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

    def theoretical_distribution_a(self, k):
        """
        Theoretical in-degree distribution for "a" nodes.
        p(k) = A_a × Γ(k + α_a + γ_a) / Γ(k + α_a)
        """
        alpha_a = self.mu_a / self.lambda_a
        gamma_a = 1 + (1 /(self.Z_tilde*self.lambda_a))
        
        A_a_piece1 = (self.m_edges*(self.f_a)) / (1 + (self.mu_a*self.Z_tilde))
        A_a_piece2 = 1 #(alpha_a)/(alpha_a + gamma_a)
        A_a_piece3 = gamma_func(alpha_a + gamma_a)/gamma_func(alpha_a)
        A_a = A_a_piece1 * A_a_piece2 * A_a_piece3
        
        numerator = gamma_func(k + alpha_a)
        denominator =  gamma_func(k + alpha_a + gamma_a)
        return A_a * (numerator / denominator)
    
    def theoretical_distribution_b(self, k):
        """
        Theoretical in-degree distribution for "b" nodes.
        p(k) = A_b × Γ(k + α_b + γ_b) / Γ(k + α_b)
        """
        
        alpha_b = self.mu_b / self.lambda_b
        gamma_b = 1 + (1 /(self.Z_tilde* self.lambda_b))

        A_b_piece1 = (self.m_edges*self.f_b) / ((1 + (self.mu_b*self.Z_tilde)))
        A_b_piece2 = 1 #(alpha_b)/(alpha_b + gamma_b)
        A_b_piece3 = gamma_func(alpha_b + gamma_b)/gamma_func(alpha_b)
        A_b = A_b_piece1 * A_b_piece2 * A_b_piece3
    
        
        numerator = gamma_func(k + alpha_b)
        denominator = gamma_func(k + alpha_b + gamma_b)
        return A_b * (numerator / denominator)
    
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
    

    def fit_asymptote(self, values, fraction=0.05):

        arr = np.array(values)
        n_tail = max(1, int(len(arr) * fraction))
        tail = arr[-n_tail:]
        return tail.mean()

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

    def print_statistics(self):
        """Print network statistics."""
        in_degrees_a = [self.graph.in_degree(n) for n in self.graph.nodes() 
                       if self.node_types[n] == 'a']
        in_degrees_b = [self.graph.in_degree(n) for n in self.graph.nodes() 
                       if self.node_types[n] == 'b']
        
        # Z-factor statistics
        if len(self.z_ratios) > 0:
            mean_ratio = np.mean(self.z_ratios)
            percent_diff = (mean_ratio - 1.0) * 100
            print(f"\nZ-factor Analysis:")
            print(f"  Mean Z_empirical/Z_theoretical = {mean_ratio:.6f}")
            print(f"  Mean % difference = {percent_diff:+.4f}%")
        
        print(f"\nType 'a': {len(in_degrees_a):,} nodes ({len(in_degrees_a)/self.graph.number_of_nodes()*100:.1f}%)")
        print(f"  Mean in-degree: {np.mean(in_degrees_a):.2f}")
        print(f"  Max in-degree: {max(in_degrees_a) if in_degrees_a else 0}")
        
        print(f"\nType 'b': {len(in_degrees_b):,} nodes ({len(in_degrees_b)/self.graph.number_of_nodes()*100:.1f}%)")
        print(f"  Mean in-degree: {np.mean(in_degrees_b):.2f}")
        print(f"  Max in-degree: {max(in_degrees_b) if in_degrees_b else 0}")

if __name__ == "__main__":
    # Generate network with homophily
    net = DirectedHomophilicNetwork(
        n0=200, 
        n_nodes=5000, 
        m_edges=3, 
        h=0.8,
        f_a=0.6,
        mu_a=1,
        mu_b=1
    )
    
    print("Generating network...")
    net.generate_network()
    print("Network generated!\n")
    
    # Plot
    fig = net.plot_degree_distributions()
    plt.show()

    # After generating the network
    fig = net.plot_in_edge_asymptotes()
    plt.show()

    
    # Statistics
    net.print_statistics()