import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

class BANetwork:
    """
    Barabási-Albert network model with customizable attachment mechanism.
    """
    
    def __init__(self, n0=-2, n_nodes=100, m=1):
        """
        Initialize BA model parameters.
        
        Args:
            n0: Initial number of nodes (complete graph)
            n_nodes: Total nodes to add
            m: Number of edges each new node creates
        """
        self.n0 = n0
        self.n_nodes = n_nodes
        self.m = m
        self.graph = None
        self.attachment_history = []
    
    def linear_preferential_attachment(self, node, candidates):
        """Standard BA: probability proportional to degree."""
        degrees = np.array([self.graph.degree(c) for c in candidates])
        probs = degrees / degrees.sum()
        return np.random.choice(candidates, self.m, p=probs, replace=False)
    
    def sublinear_preferential_attachment(self, node, candidates):
        """Sublinear attachment: probability ∝ sqrt(degree)."""
        degrees = np.array([np.sqrt(self.graph.degree(c)) for c in candidates])
        probs = degrees / degrees.sum()
        return np.random.choice(candidates, self.m, p=probs, replace=False)
    
    def fitness_weighted_attachment(self, node, candidates, fitness=None):
        """
        Attachment weighted by node fitness/absorptive capacity.
        fitness: dict mapping node -> fitness value [0, 1]
        """
        if fitness is None:
            fitness = {n: 1.0 for n in self.graph.nodes()}
        
        degrees = np.array([self.graph.degree(c) for c in candidates])
        fitnesses = np.array([fitness.get(c, 1.0) for c in candidates])
        
        # Combined: degree × fitness
        weights = degrees * fitnesses
        probs = weights / weights.sum()
        return np.random.choice(candidates, self.m, p=probs, replace=False)
    
    def technological_distance_attachment(self, node, candidates, tech_vectors=None):
        """
        Attachment based on technological proximity (Jaffe-style).
        Nodes with similar technology vectors are more likely to connect.
        
        tech_vectors: dict mapping node -> feature vector
        """
        if tech_vectors is None:
            tech_vectors = {n: np.random.rand(5) for n in self.graph.nodes()}
        
        new_tech = tech_vectors.get(node, np.random.rand(5))
        
        # Cosine similarity to candidate nodes
        similarities = []
        for c in candidates:
            cand_tech = tech_vectors.get(c, np.random.rand(5))
            sim = np.dot(new_tech, cand_tech) / (
                np.linalg.norm(new_tech) * np.linalg.norm(cand_tech) + 1e-8
            )
            similarities.append(max(0, sim))  # Ensure non-negative
        
        similarities = np.array(similarities)
        degrees = np.array([self.graph.degree(c) for c in candidates])
        
        # Combined: degree × (1 + similarity)
        weights = degrees * (1 + similarities)
        probs = weights / weights.sum()
        return np.random.choice(candidates, self.m, p=probs, replace=False)
    
    def grow(self, attachment_fn='linear', **kwargs):
        """
        Grow the BA network using specified attachment mechanism.
        
        Args:
            attachment_fn: 'linear', 'sublinear', 'fitness', or 'tech_distance'
            **kwargs: arguments passed to attachment function (e.g., fitness, tech_vectors)
        """
        # Initialize with complete graph
        self.graph = nx.complete_graph(self.n0)
        self.attachment_history = []
        
        # Select attachment function
        if attachment_fn == 'linear':
            attach_func = self.linear_preferential_attachment
        elif attachment_fn == 'sublinear':
            attach_func = self.sublinear_preferential_attachment
        elif attachment_fn == 'fitness':
            attach_func = lambda n, c: self.fitness_weighted_attachment(
                n, c, fitness=kwargs.get('fitness')
            )
        elif attachment_fn == 'tech_distance':
            attach_func = lambda n, c: self.technological_distance_attachment(
                n, c, tech_vectors=kwargs.get('tech_vectors')
            )
        else:
            raise ValueError(f"Unknown attachment function: {attachment_fn}")
        
        # Add nodes one by one
        for new_node in range(self.n0, self.n0 + self.n_nodes):
            candidates = list(self.graph.nodes())
            targets = attach_func(new_node, candidates)
            
            self.graph.add_node(new_node)
            for target in targets:
                self.graph.add_edge(new_node, target)
            
            self.attachment_history.append({
                'node': new_node,
                'targets': targets,
                'avg_target_degree': np.mean([self.graph.degree(t) for t in targets])
            })
    
    def degree_distribution(self):
        """Return degree sequence and histogram."""
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        return degrees
    
    def plot(self, figsize=(12, 5)):
        """Visualize network and degree distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Network visualization (spring layout)
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        nx.draw_networkx_nodes(
            self.graph, pos, node_size=50, node_color='lightblue', ax=ax1
        )
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, ax=ax1)
        ax1.set_title(f'BA Network (n={self.graph.number_of_nodes()}, m={self.m})')
        ax1.axis('off')
        
        # Degree distribution (log-log)
        degrees = self.degree_distribution()
        degree_counts = Counter(degrees)
        ks = sorted(degree_counts.keys())
        counts = [degree_counts[k] for k in ks]
        
        ax2.scatter(ks, counts, alpha=0.6, s=100)
        ax2.set_xlabel('Degree k')
        ax2.set_ylabel('Count')
        ax2.set_title('Degree Distribution')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def stats(self):
        """Return basic network statistics."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'avg_degree': 2 * self.graph.number_of_edges() / self.graph.number_of_nodes(),
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
        }


# Example usage
if __name__ == '__main__':
    # Standard BA model
    ba = BANetwork(n0=2, n_nodes=98, m=2)
    ba.grow('sublinear')
    print("Linear BA:", ba.stats())
    
    # Sublinear attachment
    ba_sub = BANetwork(n0=2, n_nodes=98, m=2)
    ba_sub.grow('sublinear')
    print("Sublinear BA:", ba_sub.stats())
    
    # With fitness (e.g., absorptive capacity)
    fitness = {i: np.random.uniform(0.5, 1.5) for i in range(100)}
    ba_fit = BANetwork(n0=2, n_nodes=98, m=2)
    ba_fit.grow('fitness', fitness=fitness)
    print("Fitness-weighted BA:", ba_fit.stats())
    
    # With technological distance
    tech = {i: np.random.rand(5) for i in range(100)}
    ba_tech = BANetwork(n0=2, n_nodes=98, m=2)
    ba_tech.grow('tech_distance', tech_vectors=tech)
    print("Tech-distance BA:", ba_tech.stats())
    
    # Visualization
    ba.plot()
    plt.show()