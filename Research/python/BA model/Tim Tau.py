import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import beta

class BANetwork:
    """
    Barabási-Albert network with linear preferential attachment (Price model).
    PI(k_in, t) = (k_in + mu) / ((m + mu) * t)
    where t is the cumulative number of edges added so far.
    """
    
    def __init__(self, n0, n_nodes, m_edges, mu=1):
        """
        n0: initial number of nodes
        n_nodes: number of nodes to add
        m_edges: number of edges each new node creates
        mu: fitness parameter (default=1 gives standard BA)
        """
        self.n0 = n0
        self.n_nodes = n_nodes
        self.m_edges = m_edges
        self.mu = mu
        self.graph = None
    
    def preferential_attachment(self, existing_nodes, t):
        """Select m_edges target nodes based on preferential attachment."""
        in_degrees = np.array([self.graph.in_degree(n) for n in existing_nodes])
        Z = (self.m_edges + self.mu) * t
        prob = (in_degrees + self.mu) / Z     
        return np.random.choice(existing_nodes, self.m_edges, p=prob, replace=False)
    
    def generate_network(self):
        """Generate BA network."""
        self.graph = nx.DiGraph()
        
        # Initialize with n0 nodes
        for i in range(self.n0):
            self.graph.add_node(i)
        
        # Add initial edges: m*n0 edges total to satisfy E(t_init) = m*t_init
        # Each node should have approximately m incoming edges
        edges_needed = self.m_edges * self.n0
        edges_added = 0
        
        for source in range(self.n0):
            for target in range(self.n0):
                if source != target and edges_added < edges_needed:
                    self.graph.add_edge(source, target)
                    edges_added += 1
                if edges_added >= edges_needed:
                    break
            if edges_added >= edges_needed:
                break
        
        # Now add new nodes with preferential attachment
        # t_init = n0, so start counting from t = n0
        for new_node in range(self.n0, self.n0 + self.n_nodes):
            t = new_node
            existing_nodes = list(self.graph.nodes())
            
            self.graph.add_node(new_node)
            targets = self.preferential_attachment(existing_nodes, t)
            
            for target in targets:
                self.graph.add_edge(new_node, target)
    
    def theoretical_distribution(self, k):
        """
        Theoretical degree distribution:
        p(k) = B(k + mu, 2 + (mu/m)) / B(mu, 1 + (mu/m))
        where B is the Beta function
        """
        if k < 0:
            return 0
        numerator = beta(k + self.mu, 2 + (self.mu / self.m_edges))
        denominator = beta(self.mu, 1 + (self.mu / self.m_edges))
        return numerator / denominator
    
    def logarithmic_binning(self, degrees, bin_factor=1.1):
        """
        Create logarithmic bins where each bin is ~bin_factor times larger than previous.
        Returns bin centers (sqrt(kmin*kmax)) and probabilities.
        """
        degrees = np.array(degrees)
        max_degree = np.max(degrees)
        n_total = len(degrees)
        
        # Create logarithmic bins starting from 0
        bins = [0]
        current = 1
        while current <= max_degree:
            bins.append(int(current))
            current = current * bin_factor
        bins.append(int(max_degree) + 1)
        bins = sorted(list(set(bins)))  # Remove duplicates and sort
        
        bin_centers = []
        probabilities = []
        
        for i in range(len(bins) - 1):
            kmin = bins[i]
            kmax = bins[i + 1] - 1
            
            # Count degrees in this bin
            in_bin = (degrees >= kmin) & (degrees <= kmax)
            count = np.sum(in_bin)
            
            if count > 0:
                # Bin center at sqrt(kmin * kmax)
                if kmin == 0:
                    center = np.sqrt(0.5 * kmax) if kmax > 0 else 0.5
                else:
                    center = np.sqrt(kmin * kmax)
                
                # Probability = count / total
                prob = count / n_total
                
                bin_centers.append(center)
                probabilities.append(prob)
        
        return np.array(bin_centers), np.array(probabilities)
    
    def plot_degree_distribution(self, figsize=(10, 7)):
        """
        Plot in-degree distribution with logarithmic binning.
        Compares both theoretical_distribution and theoretical_distribution2.
        """
        in_degrees = [self.graph.in_degree(n) for n in self.graph.nodes()]
        
        # Get logarithmically binned data
        bin_centers, probabilities = self.logarithmic_binning(in_degrees, bin_factor=1.1)
        
        # Calculate theoretical distributions
        k_min = max(0, int(np.min(in_degrees)))
        k_max = int(np.max(in_degrees))
        k_range = np.arange(k_min, k_max + 1)
        theoretical_probs1 = np.array([self.theoretical_distribution(k) for k in k_range])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Empirical data points
        ax.scatter(bin_centers, probabilities, s=50, alpha=0.7, 
                color='blue', edgecolors='black', linewidths=0.5,
                label='Numerical simulation', zorder=3)
        
        # Theoretical curve 1 (Beta function version)
        mask1 = theoretical_probs1 > 0
        ax.plot(k_range[mask1], theoretical_probs1[mask1], 
                '-', linewidth=2.5, color='red', alpha=0.85,
                label='Theoretical (Beta version)', zorder=2)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=14)
        ax.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=14)
        
        total_nodes = self.n0 + self.n_nodes
        ax.set_title(rf'Degree distribution from Price model' + '\n' + 
                    rf'$N = {total_nodes:,}$, $m = {self.m_edges}$, $\mu = {self.mu}$',
                    fontsize=13)
        
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Generate network with Price model parameters
    ba = BANetwork(n0=10, n_nodes=10**5, m_edges=2, mu=2)
    print("Generating network...")
    ba.generate_network()
    print("Network generated!")
    
    # Plot
    print("Creating plot...")
    fig = ba.plot_degree_distribution()
    plt.show()
    
    # Stats
    in_degrees = [ba.graph.in_degree(n) for n in ba.graph.nodes()]
    print(f"\nNetwork Statistics:")
    print(f"Total nodes: {ba.graph.number_of_nodes():,}")
    print(f"Total edges: {ba.graph.number_of_edges():,}")
    print(f"Mean in-degree: {np.mean(in_degrees):.2f}")
    print(f"Median in-degree: {np.median(in_degrees):.1f}")
    print(f"Max in-degree: {max(in_degrees)}")
    print(f"Min in-degree: {min(in_degrees)}")