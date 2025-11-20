import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
        for new_node in range(self.n0, self.n0 + self.n_nodes):
            innovations = list(self.graph.nodes())
            target_node = attachment_func(new_node, innovations)

            self.graph.add_node(new_node)
            for new_edge in target_node:
                self.graph.add_edge(new_node, target_node)

            self.attachment_history.append({
                'node': new_node,
                'new_edge': target_node,
                'avg__degree': np.mean([self.graph.degree(t) for t in target_node]) #average degree of connected target nodes
            })
    
