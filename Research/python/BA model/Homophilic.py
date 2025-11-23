import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

class BANetworkHomophily:
    """"
    Barabasi-Albert network model with homophilic preferential attachment.
    Two node types (a=majority, b=minority) with tunable homophily parameter h.
    """

    def __init__(self, n0, n_nodes, m_edges, h, f_a):
        self.n0 = n0
        self.n_nodes = n_nodes
        self.m_edges = m_edges
        self.h = h
        self.f_a = f_a
        self.f_b = 1 - f_a
        self.graph = None
        self.node_types = {}
        self.C = None
        self.beta_a = None
        self.beta_b = None
        self.gamma_a_theory = None
        self.gamma_b_theory = None

    def assign_node_types(self, n_total):
        """Assign nodes randomly to groups (a=majority, b=minority)."""
        types = np.random.choice(['a', 'b'], size=n_total, p=[self.f_a, self.f_b])
        self.node_types = {node: types[node] for node in range(n_total)}
        nodes_a = np.where(types == 'a')[0]
        nodes_b = np.where(types == 'b')[0]
        return nodes_a, nodes_b

    def solve_for_c(self):
        """Numerically solve equation (12) from Karimi et al. for C."""
        h_aa = self.h
        h_bb = self.h
        h_ab = 1 - self.h
        h_ba = 1 - self.h
        f = self.f_a
        
        def equation(C):
            numerator = f * (1 + (h_aa * C) / (h_aa * C + h_ab * (2 - C)))
            denominator_b = (h_ba * C) / (h_ba * C + h_bb * (2 - C))
            return C - (numerator + (1 - f) * denominator_b)
        
        C_solution = fsolve(equation, x0=1.0)[0]
        self.C = np.clip(C_solution, 0, 2)
        return self.C

    def compute_theoretical_gamma(self):
        """Compute theoretical gamma from C using Karimi et al. equations."""
        self.solve_for_c()
        
        h_aa = self.h
        h_bb = self.h
        h_ab = 1 - self.h
        h_ba = 1 - self.h
        f = self.f_a
        
        denominator_a = h_ba * self.C + h_bb * (2 - self.C)
        denominator_b = h_aa * self.C + h_ab * (2 - self.C)
        
        self.beta_a = (f * h_aa / denominator_a) + ((1 - f) * h_ba / denominator_b)
        self.beta_b = (f * h_ab / denominator_a) + ((1 - f) * h_bb / denominator_b)
        
        self.gamma_a_theory = -(1 / self.beta_a + 1) if self.beta_a > 0 else None
        self.gamma_b_theory = -(1 / self.beta_b + 1) if self.beta_b > 0 else None

    def homophilic_preferential_attachment(self, new_node_type, innovations):
        """Homophilic PA: probability weighted by degree and group membership."""
        same_group = np.array([n for n in innovations if self.node_types[n] == new_node_type])
        diff_group = np.array([n for n in innovations if self.node_types[n] != new_node_type])
        
        probs = np.zeros(len(innovations))
        
        if len(same_group) > 0:
            same_degrees = np.array([self.graph.degree(n) for n in same_group])
            if same_degrees.sum() > 0:
                same_probs = self.h * same_degrees / same_degrees.sum()
                probs[np.isin(innovations, same_group)] = same_probs
        
        if len(diff_group) > 0:
            diff_degrees = np.array([self.graph.degree(n) for n in diff_group])
            if diff_degrees.sum() > 0:
                diff_probs = (1 - self.h) * diff_degrees / diff_degrees.sum()
                probs[np.isin(innovations, diff_group)] = diff_probs
        
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(innovations)) / len(innovations)
        
        return np.random.choice(innovations, self.m_edges, p=probs, replace=False)

    def generate_network(self):
        """Generate the network by sequential node addition with homophilic PA."""
        self.graph = nx.complete_graph(self.n0)
        self.assign_node_types(self.n0)
        
        for new_node in range(self.n0, self.n0 + self.n_nodes):
            self.node_types[new_node] = 'a' if np.random.rand() < self.f_a else 'b'
            
            innovations = list(self.graph.nodes())
            target_node = self.homophilic_preferential_attachment(self.node_types[new_node], innovations)
            
            self.graph.add_node(new_node)
            for new_edge in target_node:
                self.graph.add_edge(new_node, new_edge)

    def degree_distribution(self, node_type=None):
        """Return degree sequence, optionally filtered by node type."""
        if node_type is None:
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        else:
            degrees = [self.graph.degree(n) for n in self.graph.nodes() 
                      if self.node_types[n] == node_type]
        return np.array(degrees)

    def get_node_colors(self):
        """Return node colors by group (red=majority, blue=minority)."""
        colors = ['red' if self.node_types[n] == 'a' else 'blue' for n in self.graph.nodes()]
        return colors

    def fit_power_law(self, x, y):
        """Fit p(k) = A * k^(-gamma) to degree distribution."""
        try:
            popt, _ = curve_fit(lambda x, A, gamma: A * x**(-gamma), x, y, p0=[100, 2], maxfev=5000)
            return popt
        except:
            return None

    def plot(self, figsize=(14, 12)):
        """Plot network and degree distributions with power law fits."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        node_colors = self.get_node_colors()
        nx.draw(self.graph, ax=axes[0, 0], node_size=1, width=0.01, node_color=node_colors)
        axes[0, 0].set_title(f"Network (h={self.h}, f_a={self.f_a})")
        
        def plot_histogram(ax, degrees, color, title):
            counts, bins = np.histogram(degrees, bins=range(1, max(degrees)+2))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, counts, width=0.8, color=color, alpha=0.6)
            
            if len(counts) > 1:
                popt = self.fit_power_law(bin_centers, counts)
                if popt is not None:
                    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 100)
                    y_fit = popt[0] * x_fit**(-popt[1])
                    ax.plot(x_fit, y_fit, 'k-', linewidth=2, 
                           label=f'A={popt[0]:.2f}, γ={popt[1]:.3f}')
                    ax.legend()
            
            ax.set_xlabel("Degree")
            ax.set_ylabel("Number of Nodes")
            ax.set_title(title)
        
        degrees_a = self.degree_distribution(node_type='a')
        degrees_b = self.degree_distribution(node_type='b')
        degrees_total = self.degree_distribution()
        
        plot_histogram(axes[0, 1], degrees_a, 'red', f"Majority (a), h={self.h}")
        plot_histogram(axes[1, 0], degrees_b, 'blue', f"Minority (b), h={self.h}")
        plot_histogram(axes[1, 1], degrees_total, 'green', f"Total, h={self.h}")
        
        plt.tight_layout()
        return fig

    def stats(self):
        """Return network statistics and compare to theory."""
        degrees_a = self.degree_distribution(node_type='a')
        degrees_b = self.degree_distribution(node_type='b')
        degrees_total = self.degree_distribution()
        
        K_a = degrees_a.sum()
        K_b = degrees_b.sum()
        K_total = degrees_total.sum()
        
        print(f"\n{'='*60}")
        print(f"Results for h = {self.h}, f_a = {self.f_a}")
        print(f"{'='*60}")
        
        print(f"\nDegree Sum Check (K_a + K_b = K_total):")
        print(f"  K_a (majority): {K_a}")
        print(f"  K_b (minority): {K_b}")
        print(f"  K_total:        {K_total}")
        print(f"  K_a + K_b:      {K_a + K_b}")
        print(f"  Match: {K_a + K_b == K_total}")
        
        print(f"\nNode counts:")
        print(f"  n_a (majority): {len(degrees_a)}")
        print(f"  n_b (minority): {len(degrees_b)}")
        print(f"  n_total: {len(degrees_total)}")
        
        print(f"\nAverage degrees:")
        print(f"  <k_a>: {np.mean(degrees_a):.3f}")
        print(f"  <k_b>: {np.mean(degrees_b):.3f}")
        print(f"  <k_total>: {np.mean(degrees_total):.3f}")
        
        print(f"\nTheoretical predictions (Karimi equations):")
        print(f"  C: {self.C:.4f}")
        print(f"  β_a (majority): {self.beta_a:.4f}")
        print(f"  β_b (minority): {self.beta_b:.4f}")
        print(f"  γ_a theory: {self.gamma_a_theory:.4f}")
        print(f"  γ_b theory: {self.gamma_b_theory:.4f}")

if __name__ == "__main__":
    ba = BANetworkHomophily(n0=10, n_nodes=2000, m_edges=3, h=0.5, f_a=0.8)
    ba.compute_theoretical_gamma()
    ba.generate_network()
    
    fig = ba.plot()
    plt.show()
    
    ba.stats()