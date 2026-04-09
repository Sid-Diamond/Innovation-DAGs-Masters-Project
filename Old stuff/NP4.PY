import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import pandas as pd
import time

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

# CONFIG
data_dir = Path("runs/run_20260312_193148/data")
layout, time_steps_diagram, size_effects, save_as_pdf = 'spring', [7, 10, 15], False, True
dpi_target, dpi_default = 400, 150
scale_factor = dpi_default / dpi_target

node_size_base = [5, 4, 3]
node_alpha = [0.8, 0.8, 0.8]
edge_width = [0.4, 0.4, 0.4]
edge_alpha = [0.6, 0.6, 0.6]
title_size = [20, 20, 20]
legend_size = [9, 9, 9]
arrowsize = [1, 1, 1]
legend_edge_width = [1.2, 1.2, 1.2]
legend_edge_alpha = [0.9, 0.9, 0.9]
new_edge_width = [1.5, 1.5, 1.5]
new_edge_alpha = [1, 1, 1]
new_edge_arrowsize = [5, 5, 5]

time_steps_diagram = sorted(list(set(time_steps_diagram)))
n_snapshots = len(time_steps_diagram)

param_lists = {
    'node_size_base': node_size_base, 'node_alpha': node_alpha, 'edge_width': edge_width,
    'edge_alpha': edge_alpha, 'title_size': title_size, 'legend_size': legend_size, 'arrowsize': arrowsize,
    'legend_edge_width': legend_edge_width, 'legend_edge_alpha': legend_edge_alpha, 'new_edge_width': new_edge_width,
    'new_edge_alpha': new_edge_alpha, 'new_edge_arrowsize': new_edge_arrowsize
}
for param_name, param_list in param_lists.items():
    param_lists[param_name] = param_list + [param_list[-1]] * max(0, n_snapshots - len(param_list)) if len(param_list) < n_snapshots else param_list[:n_snapshots]

node_size_base, node_alpha, edge_width, edge_alpha, title_size, legend_size, arrowsize, legend_edge_width, legend_edge_alpha, new_edge_width, new_edge_alpha, new_edge_arrowsize = [param_lists[k] for k in param_lists.keys()]

print("Loading CSV files...")
t0 = time.time()
nodes_df = pd.read_csv(data_dir / f"network_nodes_{layout}.csv", comment='#')
edges_df = pd.read_csv(data_dir / f"network_edges_{layout}.csv", comment='#')
summary_df = pd.read_csv(data_dir / f"network_summary_{layout}.csv", comment='#')
print(f"CSV loading: {time.time() - t0:.2f}s")

summary_dict = dict(zip(summary_df['metric'], summary_df['value']))
n0, n_added, n_total, m_edges = int(summary_dict['Initial Core Nodes']), int(summary_dict['Added Nodes']), int(summary_dict['Total Nodes']), int(summary_dict['Total Edges'])
print(f"Network: n0={n0}, n_added={n_added}, n_total={n_total}, m={m_edges}")

valid_time_steps = [t for t in time_steps_diagram if n0 <= t <= n_total]
if not valid_time_steps:
    print(f"ERROR: No valid timesteps. Must be between {n0} and {n_total}.")
    exit(1)

valid_indices = [i for i, t in enumerate(time_steps_diagram) if t in valid_time_steps]
for key in param_lists:
    exec(f"{key} = [{key}[i] for i in valid_indices]")
valid_time_steps_final = [time_steps_diagram[i] for i in valid_indices]

n_snapshots_final = len(valid_time_steps_final)
n_cols = min(3, n_snapshots_final)
n_rows = int(np.ceil(n_snapshots_final / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5*n_cols*scale_factor, 6*n_rows*scale_factor), dpi=dpi_target)
axes = np.array([axes]).flatten() if n_snapshots_final == 1 else axes.flatten()

print("Computing layout on full network...")
t0 = time.time()
G_full = nx.DiGraph()
G_full.add_nodes_from(nodes_df['node_id'].values)
G_full.add_edges_from(zip(edges_df['source'].values, edges_df['target'].values))
pos_full = nx.spring_layout(G_full, seed=42, k=1/np.sqrt(G_full.number_of_nodes()), iterations=20) if layout == 'spring' else (nx.kamada_kawai_layout(G_full) if layout == 'kamada_kawai' else nx.circular_layout(G_full))
print(f"Layout computation: {time.time() - t0:.2f}s")

for snap_idx, t in enumerate(valid_time_steps_final):
    print(f"Creating snapshot {snap_idx + 1}/{n_snapshots_final} at t={t}...")
    ax = axes[snap_idx]
    size_base, alpha, ew, ea, tsize, lsize, asize, leg_ew, leg_ea, new_ew, new_ea, new_asize = node_size_base[snap_idx], node_alpha[snap_idx], edge_width[snap_idx], edge_alpha[snap_idx], title_size[snap_idx], legend_size[snap_idx], arrowsize[snap_idx], legend_edge_width[snap_idx], legend_edge_alpha[snap_idx], new_edge_width[snap_idx], new_edge_alpha[snap_idx], new_edge_arrowsize[snap_idx]

    nodes_at_t = nodes_df[nodes_df['node_id'] < t].copy()
    node_ids_at_t = set(nodes_at_t['node_id'].values)
    most_recent_node = max(node_ids_at_t)
    edges_at_t = edges_df[(edges_df['source'].isin(node_ids_at_t)) & (edges_df['target'].isin(node_ids_at_t))].copy()

    G_t = nx.DiGraph()
    G_t.add_nodes_from(node_ids_at_t)
    G_t.add_edges_from(zip(edges_at_t['source'].values, edges_at_t['target'].values))
    pos_t = {node: pos_full[node] for node in G_t.nodes()}

    all_x, all_y = [pos[0] for pos in pos_t.values()], [pos[1] for pos in pos_t.values()]
    x_min, x_max, y_min, y_max = min(all_x), max(all_x), min(all_y), max(all_y)
    pos_t[most_recent_node] = (x_min + 0.1 * (x_max - x_min), y_max - 0.1 * (y_max - y_min))

    node_colors, node_sizes = {}, {}
    for node_id in G_t.nodes():
        node_row = nodes_at_t[nodes_at_t['node_id'] == node_id].iloc[0]
        node_colors[node_id] = 'red' if node_row['node_type'] == 'Type a' else 'blue'
        node_sizes[node_id] = size_base + (G_t.in_degree(node_id) * 2 if size_effects else 0)

    regular_edges = [(u, v) for u, v in G_t.edges() if u != most_recent_node]
    if regular_edges:
        nx.draw_networkx_edges(G_t, pos_t, edgelist=regular_edges, alpha=ea, width=ew, arrows=True, arrowsize=asize, connectionstyle='arc3,rad=0.05', ax=ax, edge_color='gray')
        for artist in ax.get_lines():
            artist.set_zorder(1)

    recent_edges = [(u, v) for u, v in G_t.edges() if u == most_recent_node]
    if recent_edges:
        nx.draw_networkx_edges(G_t, pos_t, edgelist=recent_edges, alpha=new_ea, width=new_ew, arrows=True, arrowsize=new_asize, style='dashed', connectionstyle='arc3,rad=0.05', ax=ax, edge_color='black')
        for artist in ax.get_lines():
            if artist.get_linestyle() == '--':
                artist.set_zorder(5)

    for node_id in G_t.nodes():
        node_row = nodes_at_t[nodes_at_t['node_id'] == node_id].iloc[0]
        x, y = pos_t[node_id]
        facecolor = node_colors[node_id]
        edgecolor = 'dark' + facecolor if node_row['node_origin'] == 'initial_core' else facecolor
        face = facecolor if node_row['node_origin'] == 'initial_core' else 'none'
        radius = np.sqrt(node_sizes[node_id] / np.pi) / 100
        ax.add_patch(plt.Circle((x, y), radius, facecolor=face, edgecolor=edgecolor, linewidth=1.5, alpha=alpha, zorder=3))

    ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)
        spine.set_visible(True)

    ax.set_title(f'Network at Timestep t = {t:,}', fontsize=tsize, fontweight='normal', pad=10)

    type_a_core_n = len(nodes_at_t[(nodes_at_t["node_type"] == "Type a") & (nodes_at_t["node_origin"] == "initial_core")])
    type_a_added_n = len(nodes_at_t[(nodes_at_t["node_type"] == "Type a") & (nodes_at_t["node_origin"] == "added")])
    type_b_core_n = len(nodes_at_t[(nodes_at_t["node_type"] == "Type b") & (nodes_at_t["node_origin"] == "initial_core")])
    type_b_added_n = len(nodes_at_t[(nodes_at_t["node_type"] == "Type b") & (nodes_at_t["node_origin"] == "added")])

    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='red', markersize=7, markeredgecolor='darkred', markeredgewidth=1, label=f'Type a (Core) (n={type_a_core_n})', linestyle='None'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markersize=7, markeredgecolor='red', markeredgewidth=1, label=f'Type a (Added) (n={type_a_added_n})', linestyle='None'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='blue', markersize=7, markeredgecolor='darkblue', markeredgewidth=1, label=f'Type b (Core) (n={type_b_core_n})', linestyle='None'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markersize=7, markeredgecolor='blue', markeredgewidth=1, label=f'Type b (Added) (n={type_b_added_n})', linestyle='None'),
        Line2D([0], [0], color='gray', lw=1.2, marker='>', markersize=5, label=f'Existing edges (n={len(regular_edges)})', linestyle='-'),
        Line2D([0], [0], color='black', lw=leg_ew, marker='>', markersize=5, label=f'New edges (n={len(recent_edges)})', linestyle='--'),
    ]
    legend = ax.legend(handles=legend_elements, fontsize=lsize, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, frameon=True, edgecolor='black', fancybox=False, framealpha=leg_ea)
    legend.get_frame().set_linewidth(0.5)
    ax.axis('off')

for idx in range(n_snapshots_final, len(axes)):
    axes[idx].axis('off')

for col in range(1, n_cols):
    for row in range(n_rows):
        ax_idx = row * n_cols + col
        if ax_idx < len(axes):
            fig.add_artist(plt.Line2D([col/n_cols, col/n_cols], [0, 1], transform=fig.transFigure, color='lightgray', linewidth=0.8, zorder=0))

plt.tight_layout()

if save_as_pdf:
    pdf_path = Path(__file__).parent / 'network_growth_snapshots.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=dpi_target)
    print(f"PDF saved to {pdf_path}")

plt.show()