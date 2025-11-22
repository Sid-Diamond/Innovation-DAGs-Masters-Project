import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class BANetwork:
    """"
    Barabasi-Albert network model with linear preferential attachment.
    """

    def __init__(self, n0, n_nodes, m_edges):

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
        Standard BA: probability proportional to node in_degree.
        """
        in_degrees = np.array([self.graph.in_degree(n) for n in innovations])+1  #give each paper an initial citation to avoid division by zero issues
        prob = in_degrees / in_degrees.sum()
        return np.random.choice(innovations, self.m_edges, p=prob, replace=False) 
        #replace = False to avoid multiple edges from a new node connecting to the same existing node
    
    def generate_network(self, attachment_fn='linear'):
        self.graph = nx.DiGraph()
        self.attachment_history = []

        # Add initial n0 nodes with no edges
        for i in range(self.n0):
            self.graph.add_node(i)

        if attachment_fn == 'linear':
            attachment_func = self.linear_preferential_attachment

        # Add nodes one by one
        for new_node in range(self.n0, self.n0 + self.n_nodes):
            innovations = list(self.graph.nodes())
            target_node = attachment_func(innovations)
    

            self.graph.add_node(new_node)
            for new_edge in target_node:
                self.graph.add_edge(new_node, new_edge)

            self.attachment_history.append({
                'node': new_node,
                'new_edge': target_node,
                'avg__in_degree': np.mean([self.graph.in_degree(t) for t in target_node]) #average degree of connected target nodes after the new node addition
            })
    
    def in_degree_distribution(self):
        """Return degree sequence and histogram."""
        degrees = [self.graph.in_degree(n) for n in self.graph.nodes()]
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
        
        cmap = plt.get_cmap('viridis')
        node_colors = [cmap(pct) for pct in node_time_pcts]
        
        return node_colors

    def get_temporal_positions(self):
        """Position nodes by generation (x-axis) with some vertical jitter."""
        pos = {}
        for node in self.graph.nodes():
            x = node  # x-axis = node ID (time of addition)
            y = np.random.uniform(-1, 1)  # y-axis = random jitter
            pos[node] = (x, y)
        return pos

    def plot(self, figsize=(12, 5)):
        """Visualize network and in_degree distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        node_colors = self.get_node_colors()
        pos = self.get_temporal_positions()
        nx.draw(self.graph, pos=pos, ax=ax1, node_size=0.5, width=0.1,alpha =0.6, arrowsize=10, node_color=node_colors) #alpha for transparency
        ax1.set_title("Directed BA Network")
        
        in_degrees = self.in_degree_distribution()
        ax2.hist(in_degrees, bins=range(0, max(in_degrees)+2), density=False, color='blue') #the +2 is just to ensure we dont cut the top edge off
        ax2.set_title("In_Degree Distribution")
        ax2.set_xlabel("In_Degree")
        ax2.set_ylabel("Frequency")
        
        plt.tight_layout()
        return fig
    
    def stats(self):
        """Return basic network statistics."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'avg_in_degree': np.mean([self.graph.in_degree(n) for n in self.graph.nodes()]),
            'avg_out_degree': np.mean([self.graph.out_degree(n) for n in self.graph.nodes()]),
            'density': nx.density(self.graph), #ratio of actual edges to possible edges
            #clustering coefficent = number of edges between a nodes neighbours/total possible edges between neighbours
        }

if __name__ == "__main__":
    ba = BANetwork(n0=10, n_nodes=200, m_edges=5)
    ba.generate_network(attachment_fn='linear')
    
    fig = ba.plot()
    plt.show()
    
    stats = ba.stats()
    print(stats)