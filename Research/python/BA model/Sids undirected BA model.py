import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class BANetwork:
    """"
    Barabasi-Albert network model with linear preferential attachment.
    """

    def __init__(self, n0=2, n_nodes=100, m_edges=1):

        """"
        n0: initial number of nodes (complete graph)
        n_nodes: number of nodes to add
        m_edges: number of edges each new node creates
        """
        self.n0 = n0
        self.n_nodes = n_nodes
        self.m_edges = m_edges
        self.graph = None
        self.attachment_history = []

    def linear_preferential_attachment(self, innovations):
        """"
        Standard BA: probability proportional to degree.
        """
        degrees = np.array([self.graph.degree(n) for n in innovations])
        probs = degrees / degrees.sum() #rescaled array
        return np.random.choice(innovations, self.m_edges, p=probs, replace=False) 
        #replace = False to avoid multiple edges connecting a new node to the same existing node
    
    def generate_network(self,attachment_fn='linear'):

        # Initialize with complete graph
        self.graph = nx.complete_graph(self.n0)
        self.attachment_history = []

        if attachment_fn == 'linear':
            attachment_func = self.linear_preferential_attachment

        # Add nodes one by one
        for new_node in range(self.n0, self.n0 + self.n_nodes): #add the new nodes one by one
            innovations = list(self.graph.nodes())
            target_node = attachment_func(innovations)

            self.graph.add_node(new_node)
            for new_edge in target_node:
                self.graph.add_edge(new_node, new_edge)

            self.attachment_history.append({
                'node': new_node,
                'new_edge': target_node,
                'avg__degree': np.mean([self.graph.degree(t) for t in target_node]) #average degree of connected target nodes
            })
    
    def degree_distribution(self):
        """Return degree sequence and histogram."""
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        return degrees

    def get_node_colors(self):
        """
        Return node colors based on addition time as percentage of total.
        Early nodes (added first) = 0, late nodes (added last) = 1.
        Maps to colormap and returns RGB values.Map to colormap (viridis: 0=blue, 1=yellow)
        """
        
        node_time_pcts = []
        
        for node in self.graph.nodes():
            if node < self.n0:
                time_pct = 0
            else:
                time_pct = (node - self.n0) / self.n_nodes
            
            node_time_pcts.append(time_pct)
        
        cmap = cm.get_cmap('viridis')
        node_colors = [cmap(pct) for pct in node_time_pcts]
        
        return node_colors

    def plot(self, figsize=(12, 5)):
        """Visualize network and degree distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        node_colors = self.get_node_colors()
        nx.draw(self.graph, ax=ax1, node_size=1, width=0.01, node_color=node_colors)
        ax1.set_title("BA Network")
        
        degrees = self.degree_distribution()
        ax2.hist(degrees, bins=range(1, max(degrees)+2), density=False, color='blue') #the +2 is just to ensure we dont cut the top edge off
        ax2.set_title("Degree Distribution")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Frequency")
        
        plt.tight_layout()
        return fig
    
    def stats(self):
        """Return basic network statistics."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'avg_degree': 2 * self.graph.number_of_edges() / self.graph.number_of_nodes(),
            'density': nx.density(self.graph), #ratio of actual edges to possible edges
            'avg_clustering': nx.average_clustering(self.graph), #the average clustering coefficient. 
            #clustering coefficent = number of edges between a nodes neighbours/total possible edges between neighbours
        }

if __name__ == "__main__":
    ba = BANetwork(n0=5, n_nodes=5000, m_edges=5)
    ba.generate_network(attachment_fn='linear')
    
    fig = ba.plot()
    plt.show()
    
    stats = ba.stats()
    print(stats)