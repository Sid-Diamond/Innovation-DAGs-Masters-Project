import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

data_dir = Path("runs/Scary Normalisation Blow up/data")

# ==================== CONFIG SECTION ====================
CONFIG = {
    'data_dir': data_dir,
    'figsize': (14, 6),
    'dpi': 300,
    'normalized_axis': True,
    'up_to_k': 300,
    'auto_scale_y': True,
    'ylim': [0.8, 1.2],
    'line_thickness': 2.5,
    'line_opacity': 0.8,
    'point_size': 0.2,
    'point_opacity': 1,
    'n_yticks': 5,
    'title_size': 11,
    'legend_size': 20,
    'xlabel_size': 13,
    'ylabel_size': 13,
    'xtick_size': 10,
    'ytick_size': 10,
    'num_points': False,
}
# =========================================================

def parse_csv_metadata(filepath):
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'): break
            line = line.strip('# \n')
            if 'Normalization constant' in line: metadata['title'] = line
            elif 'Node type:' in line: metadata['node_type'] = line.split(':')[1].strip()
            elif 'p0:' in line and 'alpha:' in line and 'gamma:' in line:
                parts = line.split(',')
                metadata['p0'] = float(parts[0].split(':')[1].strip())
                metadata['alpha'] = float(parts[1].split(':')[1].strip())
                metadata['gamma'] = float(parts[2].split(':')[1].strip())
            elif 'A_asymptotic:' in line: metadata['A_asymptotic'] = float(line.split(':')[1].strip())
            elif 'Generated:' in line: metadata['generated'] = line.split(':')[1].strip()
    return metadata

def plot_normalization_combined(config):
    data_dir = config['data_dir']
    figsize = config['figsize']
    dpi = config['dpi']
    normalized_axis = config['normalized_axis']
    up_to_k = config['up_to_k']
    auto_scale_y = config['auto_scale_y']
    ylim = config['ylim']
    line_thickness = config['line_thickness']
    line_opacity = config['line_opacity']
    point_size = config['point_size']
    point_opacity = config['point_opacity']
    n_yticks = config['n_yticks']
    title_size = config['title_size']
    legend_size = config['legend_size']
    xlabel_size = config['xlabel_size']
    ylabel_size = config['ylabel_size']
    xtick_size = config['xtick_size']
    ytick_size = config['ytick_size']
    num_points = config['num_points']
    
    file_patterns = ['A_normalization_type_a.csv', 'A_normalization_type_b.csv']
    existing_files = [f for f in [data_dir / p for p in file_patterns] if f.exists()]
    
    if not existing_files:
        print(f"No A_normalization files found in {data_dir}"); return
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)
    ax_right.yaxis.set_visible(False)
    
    all_metadata = {}
    data_dict = {}
    all_y_vals = []
    scale_factor = 1
    exponent = 0
    
    for data_file in existing_files:
        try:
            print(f"Processing: {data_file.name}")
            metadata = parse_csv_metadata(data_file)
            node_type = metadata['node_type']
            all_metadata[node_type] = metadata
            df = pd.read_csv(data_file, comment='#')
            df = df[df['k'] <= up_to_k]
            
            if num_points and num_points > 0:
                indices = np.linspace(0, len(df) - 1, num_points, dtype=int)
                df = df.iloc[indices].reset_index(drop=True)
            
            data_dict[node_type] = (df, metadata)
            print(f"  ✓ Parsed: p0={metadata['p0']:.6f}, α={metadata['alpha']:.6f}, γ={metadata['gamma']:.6f}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
    
    if normalized_axis and auto_scale_y:
        for node_type in data_dict:
            df, metadata = data_dict[node_type]
            y_vals = df['A_k'] / metadata['A_asymptotic']
            all_y_vals.extend(y_vals.values)
        if all_y_vals:
            all_y_vals = np.array(all_y_vals)
            all_y_vals = all_y_vals[np.isfinite(all_y_vals)]
            
            if len(all_y_vals) > 0:
                y_min, y_max = np.min(all_y_vals), np.max(all_y_vals)
                y_center = 1.0
                y_margin = max(abs(y_min - y_center), abs(y_max - y_center)) * 1.2
                ylim = [y_center - y_margin, y_center + y_margin]
                max_abs_val = max(abs(y_min - 1.0), abs(y_max - 1.0))
                if max_abs_val > 1e-10:
                    exponent = int(np.floor(np.log10(max_abs_val)))
                    scale_factor = 10 ** exponent
                else:
                    scale_factor = 1
                    exponent = 0
            else:
                scale_factor = 1
                exponent = 0
    
    for ax, (node_type, color) in [(ax_left, ('a', 'red')), (ax_right, ('b', 'blue'))]:
        if node_type not in data_dict: continue
        df, metadata = data_dict[node_type]
        
        if normalized_axis:
            y_vals = (df['A_k'] / metadata['A_asymptotic'] - 1.0) / scale_factor + 1.0
            label = f"Type '{node_type}'"
            
            valid_mask = np.isfinite(y_vals)
            explosion_idx = np.where(~valid_mask)[0]
            
            if len(explosion_idx) > 0:
                first_explosion = explosion_idx[0]
                print(f"  ⚠ Type '{node_type}': Data explosion detected at k={df.iloc[first_explosion]['k']}")
                df = df[:first_explosion]
                y_vals = y_vals[:first_explosion]
                print(f"    → Truncating plot to k={df.iloc[-1]['k']}")
            
        else:
            y_vals = df['A_k']
            label = f"Type '{node_type}'"
            dark_color = 'darkred' if node_type == 'a' else 'darkblue'
            ax.axhline(metadata['A_asymptotic'], linestyle='--', color=dark_color, linewidth=2, alpha=0.7)
        
        ax.plot(df['k'], y_vals, marker='o', color=color, linewidth=line_thickness, markersize=point_size, 
               alpha=line_opacity, label=label, zorder=3)
        ax.scatter(df['k'], y_vals, s=point_size**2*10, color=color, alpha=point_opacity, zorder=4)
        
        A_value = metadata['A_asymptotic']
        legend_label = f"$A_{{{node_type}}}(k=1) = p_{{{node_type},0}}\\frac{{\\Gamma(\\alpha_{node_type}+\\gamma_{node_type})}}{{\\Gamma(\\alpha_{node_type})}} = {A_value:.4f}$"
        
        if node_type == 'a':
            ax.text(0.98, 0.97, legend_label, transform=ax.transAxes, fontsize=legend_size, 
                   verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
        else:
            ax.text(0.98, 0.03, legend_label, transform=ax.transAxes, fontsize=legend_size, 
                   verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
        
        ax.set_xlabel('$k$', fontsize=xlabel_size, fontweight='bold')
        ax.tick_params(axis='x', labelsize=xtick_size)
        ax.tick_params(axis='y', labelsize=ytick_size)
        ax.ticklabel_format(style='plain', axis='both')
        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.6, zorder=0)
    
    if normalized_axis:
        if scale_factor != 1:
            ylabel = f'$\\left(A_{{\\mathrm{{computed}}}}/A_{{\\mathrm{{asymptotic}}}}\\right) \\times 10^{{{exponent}}}$'
        else:
            ylabel = f'$A_{{\\mathrm{{computed}}}}/A_{{\\mathrm{{asymptotic}}}}$'
        ax_left.axhline(1.0, linestyle='--', color='black', linewidth=2, alpha=0.5, zorder=2)
        ax_right.axhline(1.0, linestyle='--', color='black', linewidth=2, alpha=0.5, zorder=2)
    else:
        ylabel = '$A(k)$'
    
    ax_left.set_ylabel(ylabel, fontsize=ylabel_size, fontweight='bold')
    
    if ylim is not None:
        scaled_ylim = [(y - 1.0) / scale_factor + 1.0 for y in ylim]
        ax_left.set_ylim(scaled_ylim)
        yticks = np.linspace(scaled_ylim[0], scaled_ylim[1], n_yticks)
        ax_left.set_yticks(yticks)
    else:
        yticks = ax_left.get_yticks()
        ax_left.set_yticks(np.linspace(ax_left.get_ylim()[0], ax_left.get_ylim()[1], n_yticks))
    
    ax_right.spines['left'].set_visible(True)
    ax_right.spines['left'].set_linewidth(0.5)
    ax_right.spines['left'].set_color('gray')
    
    plt.tight_layout()
    
    suffix = "_normalized" if normalized_axis else ""
    script_dir = Path(__file__).parent
    output_path = script_dir / f"A_normalization_combined{suffix}.pdf"
    plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}\n")
    plt.close()

def main():
    if not CONFIG['data_dir'].exists(): 
        print(f"ERROR: Data directory not found: {CONFIG['data_dir']}"); return
    plot_normalization_combined(CONFIG)

if __name__ == "__main__": main()