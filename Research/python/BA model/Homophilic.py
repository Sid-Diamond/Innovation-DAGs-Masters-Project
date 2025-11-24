import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import stats

class BANetworkHomophily:
    """
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
            t1 = f
            t2 = (f * h_aa*C) / (h_aa * C + h_ab * (2 - C))
            t3 = (1 - f) * h_ba * C / (h_bb * (2 - C) + h_ba * C)
            return C - (t1 + t2 + t3)
        
        C_solution = fsolve(equation, x0=1.0)[0]
        self.C = np.clip(C_solution, 0, 2)
        
        if self.C == 0 or self.C == 2:
            print("Warning: C clipped to boundary (0 or 2). Check validity.")
        
        return self.C

    def compute_gamma_a(self):
        """Compute theoretical gamma_a from C using Karimi et al. equations."""
        self.solve_for_c()
        
        h_aa = self.h
        h_ab = 1 - self.h
        h_ba = 1 - self.h
        h_bb = self.h
        f_a = self.f_a
        f_b = 1 - f_a
        
        t1 = (f_a * h_aa) / (h_aa * self.C + h_ab * (2 - self.C))
        t2 = (f_b * h_ba) / (h_ba * self.C + h_bb * (2 - self.C))
        self.beta_a = t1 + t2
        
        self.gamma_a_theory = -(1 / self.beta_a + 1)
        return self.gamma_a_theory

    def compute_gamma_b(self):
        """Compute theoretical gamma_b from C using Karimi et al. equations."""
        if self.C is None:
            self.solve_for_c()
        
        h_aa = self.h
        h_ab = 1 - self.h
        h_ba = 1 - self.h
        h_bb = self.h
        f_a = self.f_a
        f_b = 1 - f_a
        
        t1 = (f_b * h_bb) / (h_ba * self.C + h_bb * (2 - self.C))
        t2 = (f_a * h_ab) / (h_aa * self.C + h_ab * (2 - self.C))
        self.beta_b = t1 + t2
        
        self.gamma_b_theory = -(1 / self.beta_b + 1)
        return self.gamma_b_theory

    def homophilic_preferential_attachment(self, new_node_type, innovations):
        """
        Homophilic PA following Karimi et al. equation (1).
        Returns m nodes weighted by degree and homophily.
        """
        degrees = np.array([self.graph.degree(n) for n in innovations])
        homophilies = np.array([self.h if self.node_types[n] == new_node_type 
                                else (1 - self.h) 
                                for n in innovations])
        
        numerators = homophilies * degrees
        denominator = numerators.sum()
        
        if denominator > 0:
            probs = numerators / denominator
        else:
            probs = np.ones(len(innovations)) / len(innovations)
            # Avoid division by zero if all degrees are zero and have a uniform prob distribution
        
        m = min(self.m_edges, len(innovations))
        return np.random.choice(innovations, m, p=probs, replace=False)

    def generate_network(self):
        """Generate network by sequential node addition with homophilic PA."""
        self.graph = nx.complete_graph(self.n0)
        self.assign_node_types(self.n0)
        
        for new_node in range(self.n0, self.n0 + self.n_nodes):
            new_type = 'a' if np.random.rand() < self.f_a else 'b'
            self.node_types[new_node] = new_type
            
            innovations = list(self.graph.nodes())
            target_nodes = self.homophilic_preferential_attachment(new_type, innovations)
            
            self.graph.add_node(new_node)
            for target in target_nodes:
                self.graph.add_edge(new_node, target)

    def degree_distribution(self, node_type=None):
        """Return degree sequence, optionally filtered by node type."""
        if node_type is None:
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        else:
            degrees = [self.graph.degree(n) for n in self.graph.nodes() 
                    if self.node_types[n] == node_type]
        return np.array(degrees)
    
    def get_node_colors(self):
        """Return node colors by group (red=a, blue=b)."""
        colors = ['red' if self.node_types[n] == 'a' else 'blue' 
                  for n in self.graph.nodes()]
        return colors

    def find_optimal_kmin(self,degrees):
        """Find optimal k_min by maximizing KS test p-value."""
        
        degrees = np.asarray(degrees)
        unique_degrees = np.unique(degrees)
        
        best_kmin = unique_degrees[0]
        best_pvalue = -1
        worse_count = 0
        
        for k in unique_degrees:
            degrees_filtered = degrees[degrees >= k]
            n = len(degrees_filtered)
            gamma = 1 + n / np.sum(np.log(degrees_filtered / k))
            
            pvalue = stats.kstest(degrees_filtered, 
                                lambda x: 1 - (k / x)**(gamma - 1))[1]
            
            if pvalue > best_pvalue:
                best_pvalue = pvalue
                best_kmin = k
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= 3:
                    break
        
        return best_kmin

    def fit_powerlaw_mle(self, degrees):
        """
        Fit power law exponent using MLE: γ = -(1 + n / Σ(ln(k_i / k_min)))
        Returns dict with gamma_mle, k_min, n_above_kmin, ks_statistic, ks_pvalue.
        """
        degrees = np.asarray(degrees)
        k_min = self.find_optimal_kmin(degrees)
        degrees_filtered = degrees[degrees >= k_min]
        n = len(degrees_filtered)
        
        # MLE gives positive exponent; negate for p(k) ~ k^γ convention
        gamma_mle = -(1 + n / np.sum(np.log(degrees_filtered / k_min)))
        
        degrees_sorted = np.sort(degrees_filtered)
        empirical_cdf = np.arange(1, n + 1) / n
        # CDF: F(k) = 1 - (k_min/k)^(|gamma|-1)
        theoretical_cdf = 1 - (k_min / degrees_sorted)**(-gamma_mle - 1)
        theoretical_cdf = np.clip(theoretical_cdf, 0, 1)
        
        ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        ks_pvalue = stats.kstest(degrees_filtered, 
                                lambda x: 1 - (k_min / x)**(-gamma_mle - 1))[1]
        
        return {
            'gamma_mle': gamma_mle,
            'k_min': k_min,
            'n_above_kmin': n,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue
        }

    def plot(self, figsize_network=(10, 10), figsize_dists=(14, 5)):
        """Plot network and degree distributions in separate figures."""
        
        # Figure 1: Network visualization
        fig_net, ax_net = plt.subplots(figsize=figsize_network)
        node_colors = self.get_node_colors()
        nx.draw(self.graph, ax=ax_net, node_size=1, width=0.05, alpha=0.7,
                node_color=node_colors, with_labels=False)
        ax_net.set_title(f"Network (h={self.h}, f_a={self.f_a})", fontsize=14, fontweight='bold')
        fig_net.tight_layout()
        
        # Figure 2: Degree distributions side by side
        fig_dist, axes = plt.subplots(1, 2, figsize=figsize_dists)
        self._plot_degree_distribution(axes[0], 'a', 'red', 'Majority (a)')
        self._plot_degree_distribution(axes[1], 'b', 'blue', 'Minority (b)')
        fig_dist.tight_layout()
        
        return fig_net, fig_dist

    def _plot_degree_distribution(self, ax, node_type, color, title):
        """Helper: plot histogram with log-log power law fit."""
        degrees = self.degree_distribution(node_type=node_type)
        
        counts, bins = np.histogram(degrees, bins=range(1, max(degrees) + 2))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot histogram
        ax.bar(bin_centers, counts, width=0.8, color=color, alpha=0.6, label='Data')
        
        # MLE fit
        mle_result = self.fit_powerlaw_mle(degrees)
        gamma_mle = mle_result['gamma_mle']
        k_min = mle_result['k_min']
        
        # Plot power law line in log-log space
        x_fit = np.logspace(np.log10(k_min), np.log10(degrees.max()), 100)
        
        # Normalized power law: p(k) = A * k^gamma where A = -(gamma+1)*k_min^(-(gamma+1))
        A_mle = -(gamma_mle + 1) * k_min**(-(gamma_mle + 1))
        y_fit_mle = A_mle * x_fit**(gamma_mle)
        
        ax.plot(x_fit, y_fit_mle, 'k--', linewidth=2, 
            label=f'γ={gamma_mle:.3f} (k_min={k_min}, p={mle_result["ks_pvalue"]:.3f})')
        
        # Set log-log axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.legend()
        ax.set_xlabel("Degree k")
        ax.set_ylabel("Count")
        ax.set_title(title)

    def stats(self):
        """Print network statistics and compare empirical vs. theory."""
        degrees_a = self.degree_distribution(node_type='a')
        degrees_b = self.degree_distribution(node_type='b')
        degrees_total = self.degree_distribution()
        
        
        mle_a = self.fit_powerlaw_mle(degrees_a)
        mle_b = self.fit_powerlaw_mle(degrees_b)
        
        print(f"Results for h = {self.h}, f_a = {self.f_a}")
        print(f"\nNode counts: n_a={len(degrees_a)}, n_b={len(degrees_b)}, total={len(degrees_total)}")
        print(f"Mean degrees: <k_a>={np.mean(degrees_a):.3f}, <k_b>={np.mean(degrees_b):.3f}")
        print(f"\nTheoretical (Karimi): C={self.C:.4f}")
        
        print(f"\nGroup a:")
        print(f"  MLE:     γ={mle_a['gamma_mle']:.4f} (k_min={mle_a['k_min']}, p={mle_a['ks_pvalue']:.4f})")
        print(f"  Theory:  γ={self.gamma_a_theory:.4f}")
        
        print(f"\nGroup b:")
        print(f"  MLE:     γ={mle_b['gamma_mle']:.4f} (k_min={mle_b['k_min']}, p={mle_b['ks_pvalue']:.4f})")
        print(f"  Theory:  γ={self.gamma_b_theory:.4f}")

if __name__ == "__main__":
    ba = BANetworkHomophily(n0=10, n_nodes=3000, m_edges=3, h=0.8, f_a=0.8)
    
    ba.compute_gamma_a()
    ba.compute_gamma_b() 
    ba.generate_network()
    
    fig_net, fig_dist = ba.plot()
    plt.show()
    
    ba.stats()