import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import re

class CSN2DVisFromCSV:
    def __init__(self):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.major.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 0.8
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

    def _create_csn_colormap(self):
        base = plt.colormaps['viridis']
        viridis_colors = base(np.linspace(0.2, 1.0, 256))
        return ListedColormap(viridis_colors)

    def _prepare_data_for_csn_colormap(self, data_grid, valid_mask):
        data_prepared = data_grid.copy()
        data_prepared[~valid_mask] = 0.0
        return data_prepared

    def _add_not_assessed_region_and_line(self, ax, m_values):
        m_min, m_max = float(m_values.min()), float(m_values.max())
        verts = [(m_min, 0.0), (m_max, 0.0), (m_max, m_max), (m_min, m_min)]
        light_grey = (0.85, 0.85, 0.85)
        poly = mpatches.Polygon(verts, closed=True, facecolor=light_grey, edgecolor='none', zorder=20)
        ax.add_patch(poly)
        m_boundary = np.linspace(m_min, m_max, 400)
        ax.plot(m_boundary, m_boundary, color='black', linewidth=1.5, alpha=0.9, zorder=25)
        return mpatches.Patch(facecolor=light_grey, edgecolor='black', linewidth=0.8, label='Not assessed')

    def _upsample_grid(self, m_values, n0_values, data, factor=4):
        valid = ~np.isnan(data)
        if not np.any(valid):
            return m_values, n0_values, data
        f = RegularGridInterpolator(
            (n0_values, m_values),
            np.nan_to_num(data, nan=np.nanmean(data[valid])),
            bounds_error=False,
            fill_value=np.nan,
        )
        m_fine = np.linspace(m_values.min(), m_values.max(), factor * len(m_values))
        n0_fine = np.linspace(n0_values.min(), n0_values.max(), factor * len(n0_values))
        Mf, N0f = np.meshgrid(m_fine, n0_fine)
        pts = np.stack([N0f.ravel(), Mf.ravel()], axis=-1)
        data_fine = f(pts).reshape(N0f.shape)
        return m_fine, n0_fine, data_fine
    

    def _extract_pc_from_csv(self, csv_path):
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Only scan comment header
                if not line.startswith('#'):
                    break

                # Match patterns like:
                # "# frac_nodes grid for p_c=0.2"
                match = re.search(r'p_c\s*=\s*([0-9]*\.?[0-9]+)', line, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None

        return None


    def _parse_csv_grid(self, csv_path):
        df = pd.read_csv(csv_path, comment='#', skipinitialspace=True)

        # --- Enforce numeric grid coordinates ---
        df['m'] = pd.to_numeric(df['m'], errors='coerce')
        df['n0'] = pd.to_numeric(df['n0'], errors='coerce')

        # Drop malformed rows (prevents KeyError: nan)
        df = df.dropna(subset=['m', 'n0'])

        # These are conceptually integers → remove float ambiguity
        df['m'] = df['m'].astype(int)
        df['n0'] = df['n0'].astype(int)

        # --- Identify metric column ---
        metric_cols = [c for c in df.columns if c.lower() not in ['m', 'n0', 'valid']]
        if not metric_cols:
            raise ValueError(f"No metric column found in {csv_path}")
        metric_name = metric_cols[0]

        # --- Build grid axes ---
        m_values = np.sort(df['m'].unique())
        n0_values = np.sort(df['n0'].unique())

        data_grid = np.full((len(n0_values), len(m_values)), np.nan)
        valid_mask = np.zeros((len(n0_values), len(m_values)), dtype=bool)

        m_idx_map = {m: i for i, m in enumerate(m_values)}
        n0_idx_map = {n0: i for i, n0 in enumerate(n0_values)}

        invalid_points_m = []
        invalid_points_n0 = []

        # --- Populate grid ---
        for _, row in df.iterrows():
            m_val = row['m']
            n0_val = row['n0']

            # Defensive guard (shouldn't trigger after dropna, but safe)
            if pd.isna(m_val) or pd.isna(n0_val):
                continue

            i = n0_idx_map[n0_val]
            j = m_idx_map[m_val]

            is_valid = row['valid']
            if isinstance(is_valid, str):
                is_valid = is_valid.lower() in ['true', '1', 'yes']

            valid_mask[i, j] = is_valid

            if is_valid and pd.notna(row[metric_name]):
                data_grid[i, j] = row[metric_name]

            if not is_valid:
                invalid_points_m.append(m_val)
                invalid_points_n0.append(n0_val)

        invalid_points_m = np.array(invalid_points_m)
        invalid_points_n0 = np.array(invalid_points_n0)

        # Remove invalid points inside grey (n0 <= m) region
        if invalid_points_m.size > 0:
            mask_outside_grey = invalid_points_n0 > invalid_points_m
            invalid_points_m = invalid_points_m[mask_outside_grey]
            invalid_points_n0 = invalid_points_n0[mask_outside_grey]

        return m_values, n0_values, data_grid, valid_mask, metric_name, invalid_points_m, invalid_points_n0



    def _infer_metric_type(self, metric_name, data_grid):
        if any(kw in metric_name.lower() for kw in ['frac', 'fraction', 'prob', 'probability']):
            return True
        valid_data = data_grid[~np.isnan(data_grid)]
        if valid_data.size > 0:
            if np.nanmin(valid_data) >= -0.1 and np.nanmax(valid_data) <= 1.1:
                return True
        return False

    def plot_from_csv(self, csv_path, metric_label=None, use_csn_cmap=None, figsize=(8, 7), show=True, save_path=None):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        print(f"\n📊 Loading CSV: {csv_path.name}")
        m_values, n0_values, data_grid, valid_mask, metric_name, invalid_m, invalid_n0 = self._parse_csv_grid(csv_path)
        print(f"   Metric: {metric_name}")
        print(f"   m range: [{m_values.min()}, {m_values.max()}]")
        print(f"   n0 range: [{n0_values.min()}, {n0_values.max()}]")
        print(f"   Valid points: {np.sum(valid_mask)} / {valid_mask.size}")

        pc_val = self._extract_pc_from_csv(csv_path)

        if use_csn_cmap is None:
            use_csn_cmap = self._infer_metric_type(metric_name, data_grid)
            print(f"   Inferred metric type: {'[0,1] bounded' if use_csn_cmap else 'diverging'}")

        if metric_label is None:
            if 'frac_nodes' in metric_name.lower():
                metric_label = r'$F_{\mathrm{nodes}}^*$'
            elif 'frac_edges' in metric_name.lower():
                metric_label = r'$F_{\mathrm{edges}}^*$'
            elif 'margin' in metric_name.lower():
                metric_label = r'$M = p^* - p_c$'
            else:
                metric_label = metric_name

        if 'nodes' in metric_label.lower():
            obj_label = 'Nodes'
        elif 'edges' in metric_label.lower():
            obj_label = 'Edges'
        else:
            obj_label = 'Nodes/Edges'

        data = data_grid.copy()
        data[~valid_mask] = np.nan

        m_fine, n0_fine, data_fine = self._upsample_grid(m_values, n0_values, data, factor=4)
        Mf, N0f = np.meshgrid(m_fine, n0_fine)

        fig, ax = plt.subplots(figsize=figsize)
        legend_elements = []

        if use_csn_cmap:
            cmap = self._create_csn_colormap()
            data_prep = self._prepare_data_for_csn_colormap(data_fine, ~np.isnan(data_fine))
            norm = Normalize(vmin=0.0, vmax=1.0)
            cs = ax.contourf(Mf, N0f, data_prep, levels=80, cmap=cmap, norm=norm, extend='neither')
            not_assessed_patch = self._add_not_assessed_region_and_line(ax, m_values)
            cbar = fig.colorbar(cs, ax=ax, extend='neither', ticks=[0, 0.25, 0.5, 0.75, 1.0])
            cbar.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
            legend_elements.extend([
                not_assessed_patch,
                Line2D([0], [0], color='black', linewidth=1.5, label=r'$m = n_0$'),
                mpatches.Patch(facecolor=cmap(norm(0.1)), edgecolor='black', linewidth=0.8, label=r'Valid ($p \ll 1$)'),
                mpatches.Patch(facecolor=cmap(norm(0.9)), edgecolor='black', linewidth=0.8, label=r'Valid ($p \approx 1$)'),
            ])
        else:
            max_abs = np.nanmax(np.abs(data_fine))
            max_abs = 1.0 if not np.isfinite(max_abs) or max_abs == 0 else max_abs
            levels = np.linspace(-max_abs, max_abs, 41)
            cs = ax.contourf(Mf, N0f, data_fine, levels=levels, cmap='coolwarm', extend='neither')
            not_assessed_patch = self._add_not_assessed_region_and_line(ax, m_values)
            cbar = fig.colorbar(cs, ax=ax, extend='neither')
            legend_elements.extend([
                not_assessed_patch,
                Line2D([0], [0], color='black', linewidth=1.5, label=r'$m = n_0$'),
            ])

        if invalid_m.size > 0:
            ax.scatter(invalid_m, invalid_n0, marker='x', s=40, c='red', linewidths=1.5, zorder=30)
            invalid_handle = Line2D(
                [0], [0],
                marker='x',
                linestyle='None',
                color='red',
                markersize=6,
                markeredgewidth=1.5,
                label='No valid range',
            )
            legend_elements.append(invalid_handle)

        ax.legend(handles=legend_elements, loc='upper left', fontsize=12,
                  framealpha=0.95, edgecolor='black', fancybox=False)

        title_base = rf'Fraction of {obj_label} Where $p_{{b,\infty}}(k)$ is Valid'
        if pc_val is not None:
            title_str = title_base + rf' when $P_c = {pc_val}$'
        else:
            title_str = title_base

        cbar.set_label(rf'{obj_label} Fraction', fontsize=18)
        ax.set_xlabel(r'$m$', fontsize=26)
        ax.set_ylabel(r'$n_0$', fontsize=26)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_title(title_str, fontsize=16, fontweight='normal', pad=10)
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=2000, bbox_inches='tight')
            print(f"   ✓ Saved to: {save_path}")
        if show:
            plt.show()
        return fig


def main():
    print("\n" + "=" * 70)
    print("CSN 2D Grid CSV to Contour Plot Converter")
    print("=" * 70)
    csv_files = [
        r"C:\Users\sidne\OneDrive - Imperial College London\Masters\Msci Project\runs\16.02.26\data\Play folders\frac_edges_pc0.2.csv",
        #r"C:\Users\sidne\OneDrive - Imperial College London\Masters\Msci Project\runs\16.02.26\data\frac_nodes_pc0.2.csv",
        #r"C:\Users\sidne\OneDrive - Imperial College London\Masters\Msci Project\runs\16.02.26\data\Play folders\frac_edges_pc0.2_Diamond.csv",
    ]
    vis = CSN2DVisFromCSV()
    for csv_path in csv_files:
        try:
            fig = vis.plot_from_csv(csv_path, show=True)
            print(f"   ✓ Plot generated successfully!")
        except FileNotFoundError as e:
            print(f"   ❌ File not found: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    print("\n" + "=" * 70)
    print("All plots completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()