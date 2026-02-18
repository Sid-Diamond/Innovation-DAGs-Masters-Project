import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import time

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

data_dir = Path("runs/run_20260218_191428/data")
layout = 'spring'

node_size = 0.5
node_alpha = 0.7
edge_width = 0.02
edge_alpha = 0.2
title_size = 24
legend_size = 14
arrowsize = 0.1

print("Loading CSV files...")
t0 = time.time()
nodes_df = pd.read_csv(data_dir / f"network_nodes_{layout}.csv", comment='#')
edges_df = pd.read_csv(data_dir / f"network_edges_{layout}.csv", comment='#')
summary_df = pd.read_csv(data_dir / f"network_summary_{layout}.csv", comment='#')
print(f"CSV loading: {time.time() - t0:.2f}s")

print("Building graph...")
t0 = time.time()
G = nx.DiGraph()
G.add_nodes_from(nodes_df['node_id'].values)
G.add_edges_from(zip(edges_df['source'].values, edges_df['target'].values))
print(f"Graph construction: {time.time() - t0:.2f}s")

print("Creating node color mapping...")
t0 = time.time()
node_type_map = dict(zip(nodes_df['node_id'].values, nodes_df['node_type'].values))
node_colors = np.array(['red' if node_type_map[n] == 'Type a' else 'blue' for n in G.nodes()])
print(f"Color mapping: {time.time() - t0:.2f}s")

print("Computing layout...")
t0 = time.time()
if layout == 'spring':
    # For large networks, use fewer iterations and higher k
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(G.number_of_nodes()), 
                        iterations=20)
elif layout == 'kamada_kawai':
    pos = nx.kamada_kawai_layout(G)
else:
    pos = nx.circular_layout(G)
print(f"Layout computation: {time.time() - t0:.2f}s")

print("Rendering figure...")
t0 = time.time()
fig, ax = plt.subplots(figsize=(14, 10), dpi=100)  # Reduced DPI for speed

# Draw edges FIRST (so they're behind nodes)
nx.draw_networkx_edges(G, pos, alpha=edge_alpha, width=edge_width, 
                       arrows=True, arrowsize=arrowsize,
                       connectionstyle='arc3,rad=0.05',  # Simpler curve
                       ax=ax, edge_color='gray')

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                       node_size=node_size, alpha=node_alpha, ax=ax)

print(f"Figure rendering: {time.time() - t0:.2f}s")

summary_dict = dict(zip(summary_df['metric'], summary_df['value']))

type_a = int(summary_dict['Type a Nodes'])
type_b = int(summary_dict['Type b Nodes'])
m = int(summary_dict['Total Edges'])

N = G.number_of_nodes()
ax.text(0.02, 0.98, f"{N:,} Node Network Graph", transform=ax.transAxes,
        ha='left', va='top', fontsize=title_size)

legend_elements = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor='red',
           markersize=8, alpha=node_alpha, label=f'Type a (n={type_a:,})'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='blue',
           markersize=8, alpha=node_alpha, label=f'Type b (n={type_b:,})'),
    Line2D([0], [0], color='grey', lw=1.5, marker='>', markersize=8,
           markerfacecolor='grey', markeredgecolor='grey', label=f'Edges: {m:,}')
]

legend = ax.legend(handles=legend_elements, fontsize=legend_size, loc='lower right',
                   framealpha=0.95, edgecolor='black', fancybox=False, frameon=True)
legend.get_frame().set_linewidth(0.5)

ax.axis('off')
plt.tight_layout()
plt.savefig('network_graph_optimized.png', dpi=150, bbox_inches='tight')
print("Figure saved!")
plt.show()