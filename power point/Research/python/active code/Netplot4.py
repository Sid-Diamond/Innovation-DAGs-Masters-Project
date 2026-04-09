import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
import time
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

data_dir = Path("runs/run_20260318_202157/data")
layout = 'spring'
time_steps_diagram = [9, 500]
size_effects = False
save_as_pdf = True

node_size_base = [18, 12]
node_alpha = [0.9, 0.9]
edge_width = [0.6, 0.4]
edge_alpha = [1, 1]
title_size = [28, 28]
legend_size = [13, 13]
arrowsize = [5, 5]
legend_edge_width = [1, 1]
legend_edge_alpha = [0.9, 0.9]
new_edge_width = [1.5, 1.5]
new_edge_alpha = [1, 1]
new_edge_arrowsize = [5, 5]
k_values = [2, 1.5]

time_steps_diagram = sorted(list(set(time_steps_diagram)))
n_snapshots = len(time_steps_diagram)

param_lists = {
    'node_size_base': node_size_base,
    'node_alpha': node_alpha,
    'edge_width': edge_width,
    'edge_alpha': edge_alpha,
    'title_size': title_size,
    'legend_size': legend_size,
    'arrowsize': arrowsize,
    'legend_edge_width': legend_edge_width,
    'legend_edge_alpha': legend_edge_alpha,
    'new_edge_width': new_edge_width,
    'new_edge_alpha': new_edge_alpha,
    'new_edge_arrowsize': new_edge_arrowsize,
    'k_values': k_values
}
for param_name, param_list in param_lists.items():
    if len(param_list) < n_snapshots:
        param_lists[param_name] = param_list + [param_list[-1]] * (n_snapshots - len(param_list))
    elif len(param_list) > n_snapshots:
        param_lists[param_name] = param_list[:n_snapshots]

(node_size_base,
 node_alpha,
 edge_width,
 edge_alpha,
 title_size,
 legend_size,
 arrowsize,
 legend_edge_width,
 legend_edge_alpha,
 new_edge_width,
 new_edge_alpha,
 new_edge_arrowsize,
 k_values) = [param_lists[k] for k in param_lists.keys()]

print("Loading CSV files...")
t0 = time.time()
nodes_df = pd.read_csv(data_dir / f"network_nodes_{layout}.csv", comment='#')
edges_df = pd.read_csv(data_dir / f"network_edges_{layout}.csv", comment='#')
summary_df = pd.read_csv(data_dir / f"network_summary_{layout}.csv", comment='#')
print(f"CSV loading: {time.time() - t0:.2f}s")

summary_dict = dict(zip(summary_df['metric'], summary_df['value']))
n0 = int(summary_dict['Initial Core Nodes'])
n_added = int(summary_dict['Added Nodes'])
n_total = int(summary_dict['Total Nodes'])
m_edges = int(summary_dict['Total Edges'])
print(f"Network: n0={n0}, n_added={n_added}, n_total={n_total}, m={m_edges}")

valid_time_steps = [t for t in time_steps_diagram if n0 <= t <= n_total]
if not valid_time_steps:
    print(f"ERROR: No valid timesteps. Must be between {n0} and {n_total}.")
    exit(1)

valid_indices = [i for i, t in enumerate(time_steps_diagram) if t in valid_time_steps]
(node_size_base,
 node_alpha,
 edge_width,
 edge_alpha,
 title_size,
 legend_size,
 arrowsize,
 legend_edge_width,
 legend_edge_alpha,
 new_edge_width,
 new_edge_alpha,
 new_edge_arrowsize,
 k_values) = [[param[i] for i in valid_indices] for param in [
    node_size_base,
    node_alpha,
    edge_width,
    edge_alpha,
    title_size,
    legend_size,
    arrowsize,
    legend_edge_width,
    legend_edge_alpha,
    new_edge_width,
    new_edge_alpha,
    new_edge_arrowsize,
    k_values
]]

valid_time_steps_final = [time_steps_diagram[i] for i in valid_indices]
n_snapshots_final = len(valid_time_steps_final)
n_cols = min(3, n_snapshots_final)
n_rows = int(np.ceil(n_snapshots_final / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 6 * n_rows), dpi=150)
axes = np.array([axes]).flatten() if n_snapshots_final == 1 else axes.flatten()

print("Computing layouts...")
t0 = time.time()

for snap_idx, t in enumerate(valid_time_steps_final):
    print(f"Creating snapshot {snap_idx + 1}/{n_snapshots_final} at t={t}...")
    ax = axes[snap_idx]
    size_base = node_size_base[snap_idx]
    alpha = node_alpha[snap_idx]
    ew = edge_width[snap_idx]
    ea = edge_alpha[snap_idx]
    tsize = title_size[snap_idx]
    lsize = legend_size[snap_idx]
    asize = arrowsize[snap_idx]
    leg_ew = legend_edge_width[snap_idx]
    leg_ea = legend_edge_alpha[snap_idx]
    new_ew = new_edge_width[snap_idx]
    new_ea = new_edge_alpha[snap_idx]
    new_asize = new_edge_arrowsize[snap_idx]
    k = k_values[snap_idx]

    nodes_at_t = nodes_df[nodes_df['node_id'] < t].copy()
    node_ids_at_t = set(nodes_at_t['node_id'].values)
    most_recent_node = max(node_ids_at_t)
    edges_at_t = edges_df[
        (edges_df['source'].isin(node_ids_at_t)) &
        (edges_df['target'].isin(node_ids_at_t))
    ].copy()

    G_t = nx.DiGraph()
    G_t.add_nodes_from(node_ids_at_t)
    G_t.add_edges_from(zip(edges_at_t['source'].values, edges_at_t['target'].values))

    if layout == 'spring':
        pos_t = nx.spring_layout(
            G_t,
            seed=42,
            k=k / np.sqrt(G_t.number_of_nodes()),
            iterations=50
        )
    elif layout == 'kamada_kawai':
        pos_t = nx.kamada_kawai_layout(G_t)
    else:
        pos_t = nx.circular_layout(G_t)

    all_x = [pos[0] for pos in pos_t.values()]
    all_y = [pos[1] for pos in pos_t.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pos_t[most_recent_node] = (
        x_min + 0.1 * (x_max - x_min),
        y_max - 0.1 * (y_max - y_min)
    )

    node_sizes = {
        node_id: (size_base + G_t.in_degree(node_id) * 2) if size_effects else size_base
        for node_id in G_t.nodes()
    }

    regular_edges = [(u, v) for u, v in G_t.edges() if u != most_recent_node]
    if regular_edges:
        nx.draw_networkx_edges(
            G_t, pos_t,
            edgelist=regular_edges,
            alpha=ea,
            width=ew,
            arrows=True,
            arrowsize=asize,
            connectionstyle='arc3,rad=0.05',
            ax=ax,
            edge_color='gray'
        )
        for artist in ax.get_lines():
            artist.set_zorder(1)

    recent_edges = [(u, v) for u, v in G_t.edges() if u == most_recent_node]
    if recent_edges:
        nx.draw_networkx_edges(
            G_t, pos_t,
            edgelist=recent_edges,
            alpha=new_ea,
            width=new_ew,
            arrows=True,
            arrowsize=new_asize,
            style='dashed',
            connectionstyle='arc3,rad=0.05',
            ax=ax,
            edge_color='black'
        )
        for artist in ax.get_lines():
            if artist.get_linestyle() == '--':
                artist.set_zorder(5)

    core_a = nodes_at_t[
        (nodes_at_t["node_type"] == "Type a") &
        (nodes_at_t["node_origin"] == "initial_core")
    ]["node_id"].tolist()
    added_a = nodes_at_t[
        (nodes_at_t["node_type"] == "Type a") &
        (nodes_at_t["node_origin"] == "added")
    ]["node_id"].tolist()
    core_b = nodes_at_t[
        (nodes_at_t["node_type"] == "Type b") &
        (nodes_at_t["node_origin"] == "initial_core")
    ]["node_id"].tolist()
    added_b = nodes_at_t[
        (nodes_at_t["node_type"] == "Type b") &
        (nodes_at_t["node_origin"] == "added")
    ]["node_id"].tolist()

    if core_a:
        nx.draw_networkx_nodes(
            G_t, pos_t,
            nodelist=core_a,
            node_size=[node_sizes[n] for n in core_a],
            node_color='red',
            edgecolors='darkred',
            linewidths=1.5,
            alpha=alpha,
            ax=ax,
        )
    if added_a:
        nx.draw_networkx_nodes(
            G_t, pos_t,
            nodelist=added_a,
            node_size=[node_sizes[n] for n in added_a],
            node_color='none',
            edgecolors='red',
            linewidths=1.5,
            alpha=alpha,
            ax=ax,
        )
    if core_b:
        nx.draw_networkx_nodes(
            G_t, pos_t,
            nodelist=core_b,
            node_size=[node_sizes[n] for n in core_b],
            node_color='blue',
            edgecolors='darkblue',
            linewidths=1.5,
            alpha=alpha,
            ax=ax,
        )
    if added_b:
        nx.draw_networkx_nodes(
            G_t, pos_t,
            nodelist=added_b,
            node_size=[node_sizes[n] for n in added_b],
            node_color='none',
            edgecolors='blue',
            linewidths=1.5,
            alpha=alpha,
            ax=ax,
        )

    margin = 0.1
    ax.set_xlim(x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min))
    ax.set_ylim(y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min))
    ax.set_aspect('equal', adjustable='datalim')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)
        spine.set_visible(True)

    ax.set_title(f'Network at Timestep: t = {t:,}', fontsize=tsize, fontweight='normal', pad=10)

    type_a_core_n = len(nodes_at_t[(nodes_at_t["node_type"] == "Type a") & (nodes_at_t["node_origin"] == "initial_core")])
    type_a_added_n = len(nodes_at_t[(nodes_at_t["node_type"] == "Type a") & (nodes_at_t["node_origin"] == "added")])
    type_b_core_n = len(nodes_at_t[(nodes_at_t["node_type"] == "Type b") & (nodes_at_t["node_origin"] == "initial_core")])
    type_b_added_n = len(nodes_at_t[(nodes_at_t["node_type"] == "Type b") & (nodes_at_t["node_origin"] == "added")])

    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='red',  markersize=7, markeredgecolor='darkred', markeredgewidth=1, label=f'Type a (Initial) (n$_{{0}}$={type_a_core_n})', linestyle='None'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markersize=7, markeredgecolor='red',     markeredgewidth=1, label=f'Type a (Added) (n={type_a_added_n})',      linestyle='None'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='blue', markersize=7, markeredgecolor='darkblue',markeredgewidth=1, label=f'Type b (Initial) (n$_{{0}}$={type_b_core_n})', linestyle='None'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markersize=7, markeredgecolor='blue',    markeredgewidth=1, label=f'Type b (Added) (n={type_b_added_n})',      linestyle='None'),
        Line2D([0], [0], color='gray',  lw=1.2,   marker='>', markersize=5, label=f'Existing edges (m$_{{\\mathrm{{tot}}}}$ = {len(regular_edges)})', linestyle='-'),
        Line2D([0], [0], color='black', lw=leg_ew,marker='>', markersize=5, label=f'New edges (m={len(recent_edges)})', linestyle='--'),
    ]

    legend = ax.legend(
        handles=legend_elements,
        fontsize=lsize,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        ncol=2,
        frameon=True,
        edgecolor='black',
        fancybox=False,
        framealpha=leg_ea
    )
    legend.get_frame().set_linewidth(0.5)
    ax.axis('off')

for idx in range(n_snapshots_final, len(axes)):
    axes[idx].axis('off')

for col in range(1, n_cols):
    for row in range(n_rows):
        ax_idx = row * n_cols + col
        if ax_idx < len(axes):
            fig.add_artist(
                plt.Line2D(
                    [col / n_cols, col / n_cols],
                    [0, 1],
                    transform=fig.transFigure,
                    color='lightgray',
                    linewidth=0.8,
                    zorder=0
                )
            )

plt.tight_layout()

if save_as_pdf:
    script_dir = Path(__file__).parent
    pdf_path = script_dir / 'network_growth_snapshots.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

plt.show()