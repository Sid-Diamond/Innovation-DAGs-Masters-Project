import numpy as np
import networkx as nx
import matplotlib
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.patches as mpatches
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
from scipy.special import gamma as gamma_func, betaln
from typing import Dict, Tuple
import time
from scipy.optimize import minimize
from pathlib import Path
from matplotlib.patches import Patch
import datetime
import json
import pandas as pd

class FileManager:

    def __init__(self, config: dict, base: str = "runs"):
        self.config = config
        self.base = Path(base)
        self.base.mkdir(exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base / f"run_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)  
        
        self.data_dir = self.run_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self._save_metadata()

    def finalize_metadata(self):
        """Update metadata with total runtime."""
        total_time = time.time() - self.start_time
        
        metadata_path = self.run_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        metadata["total_runtime_seconds"] = total_time
        metadata["total_runtime_formatted"] = f"{total_time:.2f}s ({total_time/60:.2f}m)"
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nTotal runtime: {total_time:.2f}s ({total_time/60:.2f}m)")

    def _save_metadata(self):
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def save_fig(self, fig, name: str, dpi: int = 300):
        path = self.run_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    
    def save_fig_with_suffix(self, fig, name: str, suffix: str, dpi: int = 300):
        """Save figure with theory-type or parameter suffix."""
        path = self.run_dir / f"{name}_{suffix}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
   
    def export_grid_to_csv(self, data_grid, filename, metric_name, m_values=None, n0_values=None, x_values=None, x_name=None, p_c=None, metadata_dict=None):
        """Export grid data to CSV. Handles 1D (x_values) and 2D (m_values, n0_values) cases."""
        filepath = self.data_dir / f"{filename}.csv"
        if x_values is not None:
            rows = []
            for i, x_val in enumerate(x_values):
                row = {x_name: x_val}
                if isinstance(data_grid, dict):
                    for key, values in data_grid.items():
                        row[key] = values[i]
                else:
                    row[metric_name] = data_grid[i]
                rows.append(row)
            df = pd.DataFrame(rows)
        else:
            rows = []
            for i, n0 in enumerate(n0_values):
                for j, m in enumerate(m_values):
                    valid = (n0 > m)
                    value = data_grid[i, j] if valid else np.nan
                    rows.append({'m': m, 'n0': n0, metric_name: value, 'valid': valid})
            df = pd.DataFrame(rows)
        with open(filepath, 'w') as f:
            if metadata_dict:
                for key, val in metadata_dict.items():
                    f.write(f"# {key}: {val}\n")
            elif p_c is not None:
                f.write(f"# {metric_name} grid for p_c={p_c}\n# Valid region: n0 > m\n")
            f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
            df.to_csv(f, index=False)
            
    def path(self):
        return self.run_dir

class DirectedHomophilicNetwork:
    """Optimized directed network with homophilic preferential attachment."""

    def __init__(self, n0: int, n_nodes: int, m_edges: int, h: float, f_a: float, mu_a: float, mu_b: float, seed: int = None, build_graph: bool = True, theory_type: str = 'Diamond', power_law_params: dict = None):
        """Initialize network with configurable theory type."""
        self.n0, self.n_nodes, self.m_edges = n0, n_nodes, m_edges
        self.h, self.f_a, self.f_b = h, f_a, 1 - f_a
        self.mu = {'a': mu_a, 'b': mu_b}
        self.seed = seed
        self.build_graph = build_graph
        self.theory_type = theory_type
        self.power_law_params = power_law_params or {}
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.lambda_a = h * f_a + (1 - f_a) * (1 - h)
        self.lambda_b = h * self.f_b + (1 - self.f_b) * (1 - h)
        self.lambda_ = {'a': self.lambda_a, 'b': self.lambda_b}
        
        self.graph = None
        self.node_types = None
        self.edge_evolution = []
        self.g_a, self.g_b, self.g_b_empirical, self.Z_factor, self.Z_tilde = (None, None, None, None, None,)
        self.in_degrees = None
        self.in_edges_a_count = 0
        self.in_edges_b_count = 0
        self._cdf_cache: dict = {}

    def _fit_asymptotes(self, fraction: float = 0.05):
        """Fit asymptotic g values (Diamond only)."""
        if self.theory_type != 'Diamond':
            return
        mean_deg_a = np.array([d['in_edges_a'] / d['t'] for d in self.edge_evolution])
        mean_deg_b = np.array([d['in_edges_b'] / d['t'] for d in self.edge_evolution])
        n_tail = max(1, int(len(mean_deg_a) * fraction))
        g_a = mean_deg_a[-n_tail:].mean()
        g_b_empirical = mean_deg_b[-n_tail:].mean()
        g_b = self.m_edges - g_a
        self._compute_theoretical_params(g_a, g_b, g_b_empirical)

    def _get_params(self, node_type: str) -> Dict[str, float]:
        """Get all distribution parameters for a node type."""
        lambda_x = self.lambda_[node_type]
        mu_x = self.mu[node_type]

        alpha = mu_x / lambda_x
        gamma = 1 + 1 / (self.Z_tilde * lambda_x)
        p0 = 1 / (1 + mu_x * self.Z_tilde)
        A = p0 * gamma_func(alpha + gamma) / gamma_func(alpha)

        return {'alpha': alpha, 'gamma': gamma, 'p0': p0, 'A': A}

    def _compute_theoretical_params(self, g_a: float, g_b: float, g_b_empirical: float):
        """Compute Z̃ using asymptotic g values."""
        self.g_a = g_a
        self.g_b = g_b
        self.g_b_empirical = g_b_empirical 
        self.Z_factor = (
            self.g_a * self.lambda_a
            + self.g_b * self.lambda_b
            + self.f_a * self.mu['a']
            + self.f_b * self.mu['b']
        )
        self.Z_tilde = self.m_edges / self.Z_factor

    def assign_node_type(self) -> str:
        """Randomly assign node type based on f_a."""
        return 'a' if np.random.rand() < self.f_a else 'b'

    def homophilic_preferential_attachment(self, n_nodes_so_far: int) -> np.ndarray:
        """Select m_edges targets using cached in-degrees."""
        in_deg = self.in_degrees[:n_nodes_so_far]
        types = self.node_types[:n_nodes_so_far]

        lambda_vals = np.where(types == 0, self.lambda_a, self.lambda_b)
        mu_vals = np.where(types == 0, self.mu['a'], self.mu['b'])

        probs = lambda_vals * in_deg + mu_vals
        probs = probs / probs.sum()

        return np.random.choice(n_nodes_so_far, size=self.m_edges, p=probs, replace=False)

    def generate_network(self):
        """Generate network with cached state."""
        total_nodes = self.n0 + self.n_nodes

        # Pre-allocate arrays
        self.node_types = np.empty(total_nodes, dtype=np.int8)  # 0 for 'a', 1 for 'b'
        self.in_degrees = np.zeros(total_nodes, dtype=np.int32)

        # Only build a NetworkX graph if requested
        if self.build_graph:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(range(total_nodes))
        else:
            self.graph = None

        # Initialize node types for the initial core
        for i in range(self.n0):
            self.node_types[i] = 0 if self.assign_node_type() == 'a' else 1

        # Initial random edges
        all_nodes = np.arange(self.n0)
        for source in range(self.n0):
            available = all_nodes[all_nodes != source]
            targets = np.random.choice(available, size=self.m_edges, replace=False)

            if self.build_graph:
                self.graph.add_edges_from((source, int(t)) for t in targets)

            self.in_degrees[targets] += 1

            for t in targets:
                if self.node_types[t] == 0:
                    self.in_edges_a_count += 1
                else:
                    self.in_edges_b_count += 1

        # Add nodes with preferential attachment
        for new_node in range(self.n0, total_nodes):
            self.node_types[new_node] = 0 if self.assign_node_type() == 'a' else 1

            if self.build_graph:
                self.graph.add_node(new_node)

            targets = self.homophilic_preferential_attachment(new_node)

            if self.build_graph:
                self.graph.add_edges_from((new_node, int(t)) for t in targets)

            self.in_degrees[targets] += 1

            for t in targets:
                if self.node_types[t] == 0:
                    self.in_edges_a_count += 1
                else:
                    self.in_edges_b_count += 1

            # Track evolution
            if (new_node - self.n0) % 100 == 0 or new_node == total_nodes - 1:
                self.edge_evolution.append(
                    {
                        't': new_node,
                        'in_edges_a': self.in_edges_a_count,
                        'in_edges_b': self.in_edges_b_count,
                    }
                )

        self._fit_asymptotes()

    def get_beta_for_type(self, node_type: str):
        """
        Return the theoretical beta = (p0, alpha, gamma) for a node type,
        based on the current asymptotic parameters stored in the network.
        """
        params = self._get_params(node_type)
        return np.array([params['p0'], params['alpha'], params['gamma']], dtype=float)

    def pmf_from_beta(self, k, beta) -> np.ndarray:
        """
        Core PMF: p(k | beta) with beta = (p0, alpha, gamma).
        Uses betaln for numerical stability (original approach).
        """
        p0, alpha, gamma = beta
        k = np.atleast_1d(k)
        result = np.zeros_like(k, dtype=float)
        
        zero_mask = (k == 0)
        pos_mask = (k > 0)
        
        if np.any(zero_mask):
            result[zero_mask] = p0
        
        if np.any(pos_mask):
            k_pos = k[pos_mask]
            log_ratio = betaln(k_pos, alpha + gamma) - betaln(k_pos, alpha)
            result[pos_mask] = p0 * np.exp(log_ratio)
        
        return result

    def cdf_from_beta_truncated(self, k, beta, kmin: int, kmax: int) -> np.ndarray:
        """Conditional CDF F(k | beta, kmin <= K <= kmax) on integer support (Diamond only)."""
        if self.theory_type != 'Diamond':
            raise NotImplementedError(f"CDF truncation not implemented for {self.theory_type}")
        
        k = np.atleast_1d(k)
        k_int = np.floor(k).astype(int)
        beta = np.asarray(beta, dtype=float)
        beta_key = tuple(np.round(beta, decimals=12))
        cache_key = (beta_key, int(kmin), int(kmax))
        
        if cache_key not in self._cdf_cache:
            support = np.arange(kmin, kmax + 1, dtype=int)
            pmf = self.pmf_from_beta(support, beta).astype(float)
            Z = pmf.sum()
            if Z <= 0:
                cdf_support = np.ones_like(support, dtype=float)
            else:
                pmf /= Z
                cdf_support = np.cumsum(pmf)
            self._cdf_cache[cache_key] = (support, cdf_support)
        else:
            support, cdf_support = self._cdf_cache[cache_key]
        
        F = np.zeros_like(k_int, dtype=float)
        s_min = support[0]
        s_max = support[-1]
        for i, ki in enumerate(k_int):
            if ki < s_min:
                F[i] = 0.0
            elif ki >= s_max:
                F[i] = 1.0
            else:
                F[i] = cdf_support[ki - s_min]
        return F.item() if F.size == 1 else F

    def _get_degrees(self, node_type: str) -> np.ndarray:
        """Get in-degrees for nodes of specified type."""
        type_val = 0 if node_type == 'a' else 1
        mask = self.node_types[:self.in_degrees.size] == type_val
        return self.in_degrees[mask].copy()

    def theoretical_distribution(self, k, node_type: str):
        """Theoretical in-degree distribution (dispatches by theory_type)."""
        if self.theory_type == 'Diamond':
            beta = self.get_beta_for_type(node_type)
            pmf = self.pmf_from_beta(k, beta)
            return pmf.item() if np.ndim(pmf) == 1 and pmf.size == 1 else pmf
        elif self.theory_type == 'Sterling':
            return self._pmf_Sterling(k, node_type)
        elif self.theory_type == 'power_law_2':
            return self._pmf_power_law_2(k, node_type)
        else:
            raise ValueError(f"Unknown theory_type: {self.theory_type}")

    def _compute_power_law_params(self, node_type: str):
        """
        For Sterling we reuse the Diamond asymptotic parameters
        (p0, alpha, gamma) and pre-compute the scalar prefactor A.
        For power_law_2 the original empirical-fit logic is unchanged.
        """
        if self.theory_type == 'Sterling':
            # Reuse the parameters already computed by _compute_theoretical_params
            params = self._get_params(node_type)          # needs Z_tilde to be set
            self.power_law_params[node_type] = {
                'alpha': params['alpha'],
                'gamma': params['gamma'],
                'p0':    params['p0'],
                'A':     params['A'],   # p0 * Γ(α+γ) / Γ(α)
            }

        elif self.theory_type == 'power_law_2':
            # ── original power_law_2 branch (unchanged) ──────────────────────
            degrees = self._get_degrees(node_type)
            if degrees.size == 0:
                return
            degrees_pos = degrees[degrees > 0]
            if degrees_pos.size == 0:
                return

            alpha_est = 2.0
            k0_est    = 1.0
            k_max     = int(degrees_pos.max())
            norm_est  = np.sum((np.arange(0, k_max + 1) + k0_est) ** (-alpha_est))
            self.power_law_params[node_type] = {
                'alpha': alpha_est,
                'k0':    k0_est,
                'norm':  norm_est,
            }

    def _pmf_Sterling(self, k, node_type: str) -> np.ndarray:
        """
        p(k) = A * (k + alpha)^{-gamma}
        where A = p0 * Γ(alpha + gamma) / Γ(alpha).

        Parameters come from the Diamond asymptotic fit so they are
        on the same footing as the Diamond curve.
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))

        if node_type not in self.power_law_params:
            self._compute_power_law_params(node_type)

        p = self.power_law_params[node_type]
        A, alpha, gamma = p['A'], p['alpha'], p['gamma']

        return A * (k + alpha) ** (-gamma)

    def _pmf_power_law_2(self, k, node_type: str) -> np.ndarray:
        """Power law type 2: p(k) ~ (k + k0)^(-alpha) / normalization."""
        k = np.atleast_1d(k)
        if node_type not in self.power_law_params:
            self._compute_power_law_params(node_type)
        alpha = self.power_law_params[node_type]['alpha']
        k0 = self.power_law_params[node_type]['k0']
        norm = self.power_law_params[node_type]['norm']
        result = ((k + k0) ** (-alpha)) / norm
        return result

    class NetworkVis:
        """Visualization utilities for DirectedHomophilicNetwork."""
        
        def __init__(self, parent_net: "DirectedHomophilicNetwork"):
            self.net = parent_net
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['axes.linewidth'] = 0.8
            plt.rcParams['xtick.major.width'] = 0.8
            plt.rcParams['ytick.major.width'] = 0.8

        def _logarithmic_binning(self, degrees: np.ndarray, bin_factor: float = 1.05) -> Tuple[np.ndarray, np.ndarray]:
            """Logarithmic binning with k=0 separate."""
            if len(degrees) == 0:
                return np.array([]), np.array([])
            degrees = np.array(degrees)
            max_degree, n_total = np.max(degrees), len(degrees)
            count_zero = np.sum(degrees == 0)
            bin_centers = [0.0] if count_zero > 0 else []
            probabilities = [count_zero / n_total] if count_zero > 0 else []
            if np.any(degrees > 0):
                bins = [1]
                current = 1 * bin_factor
                while current <= max_degree:
                    bins.append(int(current))
                    current *= bin_factor
                bins.append(int(max_degree) + 1)
                bins = sorted(set(bins))
                for i in range(len(bins) - 1):
                    kmin, kmax = bins[i], bins[i + 1] - 1
                    count = np.sum((degrees >= kmin) & (degrees <= kmax))
                    if count > 0:
                        center = np.sqrt(kmin * kmax)
                        bin_centers.append(center)
                        probabilities.append(count / n_total)
            return np.array(bin_centers), np.array(probabilities)

        _THEORY_STYLES = {
            'Diamond':  dict(linestyle='-',  linewidth=2.5, alpha=0.85, label='Diamond'),
            'Sterling': dict(linestyle='--', linewidth=2.0, alpha=0.85, label='Sterling'),
            'power_law_2': dict(linestyle=':',  linewidth=2.0, alpha=0.85, label='Power Law 2'),
        }

        def plot_degree_distributions_log(self, fm, theories, discretisations=10**5, figsize=(15, 6)):
            """Log-binned empirical scatter + one curve per theory on the same axes."""

            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)

            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "stix"

            y_max_fixed = 1.0        # top of axis: 10^0
            y_min_positive = None
            all_data = {}

            # ---------- FIRST PASS ----------
            for node_type, color in [('a', 'red'), ('b', 'blue')]:
                in_degrees = self.net._get_degrees(node_type)
                if len(in_degrees) == 0:
                    continue

                bin_centers, probs = self._logarithmic_binning(in_degrees)

                positive_probs = probs[probs > 0]
                if positive_probs.size > 0:
                    local_y_min = positive_probs.min()
                    y_min_positive = local_y_min if y_min_positive is None else min(y_min_positive, local_y_min)

                all_data[node_type] = {
                    "in_degrees": in_degrees,
                    "bin_centers": bin_centers,
                    "probs": probs,
                }

            if y_min_positive is not None:
                y_bottom = y_min_positive * 0.8
            else:
                y_bottom = 1e-6

            # ---------- SECOND PASS ----------
            for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
                if node_type not in all_data:
                    continue

                ax = axes[idx]
                in_degrees = all_data[node_type]["in_degrees"]
                bin_centers = all_data[node_type]["bin_centers"]
                probs = all_data[node_type]["probs"]

                csv_data = {
                    'k_bin_center': bin_centers,
                    'empirical_prob': probs
                }

                ax.scatter(
                    bin_centers,
                    probs,
                    s=50,
                    alpha=0.7,
                    color=color,
                    edgecolors='black',
                    linewidths=0.5,
                    label='Simulation',
                    zorder=3
                )

                k_max = int(np.max(in_degrees))
                if discretisations == 0:
                    k_range = np.arange(0, k_max + 1)
                else:
                    k_range = np.concatenate([[0], np.linspace(0.01, k_max, discretisations)])

                for theory in theories:
                    self.net.theory_type = theory
                    if theory == 'Diamond':
                        self.net._fit_asymptotes()
                    else:
                        self.net._compute_power_law_params(node_type)

                    theo_probs = self.net.theoretical_distribution(k_range, node_type)
                    mask = theo_probs > 0

                    base_style = dict(linestyle='-.', linewidth=2.0, alpha=0.85)
                    style = self._THEORY_STYLES.get(theory, base_style)

                    if theory == 'Diamond':
                        # node_type is 'a' or 'b', so use that in the subscript
                        style['label'] = rf'$p_{{{node_type},\infty}}(k)$'
                    else:
                        style.setdefault('label', theory)

                    ax.plot(
                        k_range[mask],
                        theo_probs[mask],
                        color='dark' + color,
                        zorder=2,
                        **style
                    )

                    theo_at_bins = self.net.theoretical_distribution(bin_centers, node_type)
                    csv_data[f'{theory}_prob'] = theo_at_bins

                ax.set_xscale('symlog', linthresh=0.1)
                #ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(left=-0.05)
                ax.set_ylim(bottom=y_bottom, top=y_max_fixed)

                ax.set_xlabel(r'Citation count', fontsize=13)
                ax.set_ylabel(r'Citation Count Probability', fontsize=13)

                n_nodes_type = len(in_degrees)
                n_nodes_str = f"{n_nodes_type:,}"
                ax.set_title(
                    rf'Type "{node_type}" paper citation distribution '
                    rf'(n={n_nodes_str})',
                    fontsize=13,
                    fontweight='normal'
                )

                legend = ax.legend(fontsize=11, loc='best', framealpha=0.9)
                for text in legend.get_texts():
                    text.set_fontfamily('serif')

                ax.grid(True, alpha=0.4, which='both', linestyle='--', linewidth=0.6)

                df = pd.DataFrame(csv_data)
                filepath = fm.data_dir / f"degree_dist_log_binned_type_{node_type}.csv"
                with open(filepath, 'w') as f:
                    f.write(f"# Log-binned degree distribution\n")
                    f.write(f"# Node type: {node_type}, n={n_nodes_str}\n")
                    f.write(f"# N: {self.net.n0 + self.net.n_nodes:,}, m: {self.net.m_edges:,}\n")
                    f.write(f"# h: {self.net.h}, f_a: {self.net.f_a}\n")
                    f.write(f"# Theories: {', '.join(theories)}\n")
                    f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                    df.to_csv(f, index=False)

                type_val = 0 if node_type == 'a' else 1
                node_mask = self.net.node_types[:self.net.in_degrees.size] == type_val
                node_ids = np.where(node_mask)[0]
                node_degrees_data = pd.DataFrame({
                    'node_id': node_ids,
                    'in_degree': in_degrees
                })
                node_filepath = fm.data_dir / f"node_degrees_type_{node_type}.csv"
                with open(node_filepath, 'w') as f:
                    f.write(f"# Raw node-level degree data\n")
                    f.write(f"# Node type: {node_type}, n={n_nodes_str}\n")
                    f.write(f"# N: {self.net.n0 + self.net.n_nodes:,}, m: {self.net.m_edges:,}\n")
                    f.write(f"# h: {self.net.h}, f_a: {self.net.f_a}\n")
                    f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                    node_degrees_data.to_csv(f, index=False)

            fig.suptitle(
                r'Citation Count Distributions in a Directed Acyclic Homophilic Network',
                fontsize=20,
                fontweight='normal',
                y=0.94
            )

            fig.tight_layout(rect=[0, 0, 1, 0.95])

            fm.save_fig(fig, "degree_dist_log_binned")
            return fig

        def plot_degree_distributions_linear(self, fm, theories, max_k_display=None, figsize=(15, 6)):
            """Discrete integer PMF scatter + one curve per theory on the same axes."""
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
                in_degrees = self.net._get_degrees(node_type)
                if len(in_degrees) == 0:
                    continue
                
                ax = axes[idx]
                unique_k, counts = np.unique(in_degrees, return_counts=True)
                empirical_pmf = counts / len(in_degrees)
                
                # CSV: empirical PMF
                csv_data = {'k': unique_k, 'empirical_pmf': empirical_pmf}
                
                ax.scatter(unique_k, empirical_pmf, s=1, alpha=0.7, color=color,
                        edgecolors='black', linewidths=1, label='Simulation', zorder=3)
                
                k_max = int(np.max(in_degrees)) if max_k_display is None else max_k_display
                k_range = np.arange(0, k_max + 1)
                
                for theory in theories:
                    self.net.theory_type = theory
                    if theory == 'Diamond':
                        self.net._fit_asymptotes()
                    else:
                        self.net._compute_power_law_params(node_type)
                    theo_probs = self.net.theoretical_distribution(k_range, node_type)
                    style = self._THEORY_STYLES.get(
                        theory, dict(linestyle='-.', linewidth=1.5, alpha=0.85, label=theory))
                    ax.plot(k_range, theo_probs, color='dark' + color, zorder=2, **style)
                
                ax.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=13)
                ax.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=13)
                ax.set_title(f'Type "{node_type}" (n={len(in_degrees)}) - LINEAR SCALE',
                            fontsize=13, fontweight='bold')
                ax.legend(fontsize=9, loc='best', framealpha=0.9)
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                
                # CSV: add full theory curves
                for theory in theories:
                    self.net.theory_type = theory
                    if theory == 'Diamond':
                        self.net._fit_asymptotes()
                    else:
                        self.net._compute_power_law_params(node_type)
                    theo_full = self.net.theoretical_distribution(k_range, node_type)
                    
                    # Merge with empirical (pad with NaN where empirical doesn't exist)
                    if len(csv_data['k']) < len(k_range):
                        k_full = k_range
                        emp_full = np.full(len(k_range), np.nan)
                        for i, k_val in enumerate(unique_k):
                            emp_full[k_val] = empirical_pmf[i]
                        csv_data = {'k': k_full, 'empirical_pmf': emp_full}
                    csv_data[f'{theory}_pmf'] = theo_full
                
                # Export CSV
                df = pd.DataFrame(csv_data)
                filepath = fm.data_dir / f"degree_dist_linear_type_{node_type}.csv"
                with open(filepath, 'w') as f:
                    f.write(f"# Linear-scale discrete degree distribution\n")
                    f.write(f"# Node type: {node_type}, n={len(in_degrees)}\n")
                    f.write(f"# N: {self.net.n0 + self.net.n_nodes}, m: {self.net.m_edges}\n")
                    f.write(f"# h: {self.net.h}, f_a: {self.net.f_a}\n")
                    f.write(f"# Theories: {', '.join(theories)}\n")
                    f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                    df.to_csv(f, index=False)
                
                # Export raw node degrees with node IDs (same as log plot to avoid duplication)
                if idx == 0:  # Only export once per run, not per plot type
                    type_val = 0 if node_type == 'a' else 1
                    node_mask = self.net.node_types[:self.net.in_degrees.size] == type_val
                    node_ids = np.where(node_mask)[0]
                    node_degrees_data = pd.DataFrame({
                        'node_id': node_ids,
                        'in_degree': in_degrees
                    })
                    node_filepath = fm.data_dir / f"node_degrees_type_{node_type}.csv"
                    # Only write if file doesn't already exist (from log plot)
                    if not node_filepath.exists():
                        with open(node_filepath, 'w') as f:
                            f.write(f"# Raw node-level degree data\n")
                            f.write(f"# Node type: {node_type}, n={len(in_degrees)}\n")
                            f.write(f"# N: {self.net.n0 + self.net.n_nodes}, m: {self.net.m_edges}\n")
                            f.write(f"# h: {self.net.h}, f_a: {self.net.f_a}\n")
                            f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                            node_degrees_data.to_csv(f, index=False)
            
            fig.suptitle(
                f'Directed Homophilic Network: N={self.net.n0 + self.net.n_nodes:,}, '
                f'm={self.net.m_edges}, h={self.net.h}, f_a={self.net.f_a}',
                fontsize=14, fontweight='bold')
            plt.tight_layout()
            fm.save_fig(fig, "degree_dist_discrete_linear")

        def plot_network_graph(self, fm: "FileManager", figsize: Tuple = (12, 12), node_size: float = 100, layout: str = 'spring', plot: bool = False):
            """Visualize network graph with node coloring by type and save network data to CSV.
            """
            if self.net.graph is None:
                print("Warning: Network graph not built. Set build_graph=True when creating network.")
                return None
            
            # Generate plot if requested
            if plot:
                fig, ax = plt.subplots(figsize=figsize)
                node_types = self.net.node_types[:self.net.graph.number_of_nodes()]
                node_colors = ['red' if nt == 0 else 'blue' for nt in node_types]
                
                pos = nx.spring_layout(self.net.graph, seed=42, k=1/np.sqrt(self.net.graph.number_of_nodes())) if layout == 'spring' else (nx.kamada_kawai_layout(self.net.graph) if layout == 'kamada_kawai' else nx.circular_layout(self.net.graph))
                
                nx.draw_networkx_nodes(self.net.graph, pos, node_color=node_colors, node_size=node_size, alpha=0.7, ax=ax)
                nx.draw_networkx_edges(self.net.graph, pos, alpha=0.2, width=0.5, arrows=True, arrowsize=10, ax=ax)
                
                legend_elements = [Patch(facecolor='red', alpha=0.7, label='Type a'), Patch(facecolor='blue', alpha=0.7, label='Type b')]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
                
                legend_text = f"N={self.net.n0 + self.net.n_nodes}, m={self.net.m_edges}, h={self.net.h}, $f_a$={self.net.f_a}"
                ax.text(0.5, 0.98, legend_text, transform=ax.transAxes, ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_title(f"Network Graph ({layout} layout)", fontsize=14, fontweight='bold', pad=20)
                ax.axis('off')
                plt.tight_layout()
                
                # Save figure
                fm.save_fig(fig, f"network_graph_{layout}")
            
            # Save network data to CSV files in data folder
            node_types = self.net.node_types[:self.net.graph.number_of_nodes()]
            
            # Nodes data
            nodes_data = []
            for node_id in self.net.graph.nodes():
                node_type = self.net.node_types[node_id]
                nodes_data.append({
                    'node_id': node_id,
                    'node_type': 'Type a' if node_type == 0 else 'Type b',
                    'in_degree': self.net.graph.in_degree(node_id),
                    'out_degree': self.net.graph.out_degree(node_id)
                })
            
            nodes_df = pd.DataFrame(nodes_data)
            nodes_filepath = fm.data_dir / f"network_nodes_{layout}.csv"
            with open(nodes_filepath, 'w') as f:
                f.write(f"# Network nodes data\n")
                f.write(f"# Layout: {layout}\n")
                f.write(f"# N: {self.net.n0 + self.net.n_nodes}, m: {self.net.m_edges}\n")
                f.write(f"# h: {self.net.h}, f_a: {self.net.f_a}\n")
                f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                nodes_df.to_csv(f, index=False)
            
            # Edges data
            edges_data = []
            for source, target in self.net.graph.edges():
                source_type = self.net.node_types[source]
                target_type = self.net.node_types[target]
                edges_data.append({
                    'source': source,
                    'target': target,
                    'source_type': 'Type a' if source_type == 0 else 'Type b',
                    'target_type': 'Type a' if target_type == 0 else 'Type b'
                })
            
            edges_df = pd.DataFrame(edges_data)
            edges_filepath = fm.data_dir / f"network_edges_{layout}.csv"
            with open(edges_filepath, 'w') as f:
                f.write(f"# Network edges data\n")
                f.write(f"# Layout: {layout}\n")
                f.write(f"# N: {self.net.n0 + self.net.n_nodes}, m: {self.net.m_edges}\n")
                f.write(f"# h: {self.net.h}, f_a: {self.net.f_a}\n")
                f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                edges_df.to_csv(f, index=False)
            
            # Summary statistics
            type_a_count = sum(1 for nt in node_types if nt == 0)
            type_b_count = sum(1 for nt in node_types if nt == 1)
            
            summary_data = {
                'metric': ['Total Nodes', 'Total Edges', 'Type a Nodes', 'Type b Nodes', 'Average In-Degree', 'Network Density'],
                'value': [
                    self.net.graph.number_of_nodes(),
                    self.net.graph.number_of_edges(),
                    type_a_count,
                    type_b_count,
                    2 * self.net.graph.number_of_edges() / self.net.graph.number_of_nodes() if self.net.graph.number_of_nodes() > 0 else 0,
                    nx.density(self.net.graph)
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_filepath = fm.data_dir / f"network_summary_{layout}.csv"
            with open(summary_filepath, 'w') as f:
                f.write(f"# Network summary statistics\n")
                f.write(f"# Layout: {layout}\n")
                f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
                summary_df.to_csv(f, index=False)
            
            return fig if plot else None

        def plot_asymptotes(self, fm: "FileManager", figsize: Tuple = (10, 6)):
            """Plot mean in-edge density with asymptotic fits (Diamond only)."""
            if self.net.theory_type != 'Diamond':
                return None
            times = np.array([d['t'] for d in self.net.edge_evolution])
            mean_deg_a = np.array([d['in_edges_a'] / d['t'] for d in self.net.edge_evolution])
            mean_deg_b = np.array([d['in_edges_b'] / d['t'] for d in self.net.edge_evolution])
            fig, ax = plt.subplots(figsize=figsize)
            for mean_deg, type_name, color in [(mean_deg_a, 'a', 'red'), (mean_deg_b, 'b', 'blue')]:
                ax.plot(times, mean_deg, label=f"Type '{type_name}' (data)", color=color, linewidth=2)
                asymptote = self.net.g_a if type_name == 'a' else self.net.g_b
                ax.axhline(asymptote, linestyle='--', alpha=0.7, color='dark' + color, label=f"Type '{type_name}' asymptote = {asymptote:.3f}", linewidth=2)
            legend_text = f"N={self.net.n0 + self.net.n_nodes:,}, m={self.net.m_edges}, h={self.net.h}"
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlabel("t (number of nodes)", fontsize=12)
            ax.set_ylabel("Mean in-degree", fontsize=12)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            fm.save_fig(fig, "asymptotes")
            csv_data = [{'t': t, 'mean_deg_a': ma, 'mean_deg_b': mb, 'asymptote_a': self.net.g_a, 'asymptote_b': self.net.g_b} for t, ma, mb in zip(times, mean_deg_a, mean_deg_b)]
            df = pd.DataFrame(csv_data)
            filepath = fm.data_dir / "asymptotes.csv"
            with open(filepath, 'w') as f:
                f.write(f"# Asymptotic in-edge density evolution\n# N: {self.net.n0 + self.net.n_nodes}, m: {self.net.m_edges}\n# h: {self.net.h}, f_a: {self.net.f_a}\n# g_a: {self.net.g_a:.6f}, g_b: {self.net.g_b:.6f}\n# Generated: {datetime.datetime.now().isoformat()}\n")
                df.to_csv(f, index=False)
            return fig

        def plot_normalization(self, fm: "FileManager", max_k: int = 25, figsize: Tuple = (15, 6)):
            """Plot theory-specific normalization. Diamond: A(k), Power Law: exponent & norm."""
            if self.net.theory_type == 'Diamond':
                return self._plot_normalization_diamond(fm, max_k, figsize)
            elif self.net.theory_type in ['Sterling', 'power_law_2']:
                return self._plot_normalization_power_law(fm, max_k, figsize)
            return None

        def _plot_normalization_diamond(self, fm: "FileManager", max_k: int, figsize: Tuple):
            """Plot A(k) normalization constant for Diamond."""
            k_values = np.arange(0, max_k + 1)
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
                params = self.net._get_params(node_type)
                A_values = [params['p0'] * (np.prod([(params['alpha'] + i) / (params['alpha'] + params['gamma'] + i) for i in range(k)]) if k > 0 else 1.0) * gamma_func(k + params['alpha'] + params['gamma']) / gamma_func(k + params['alpha']) for k in k_values]
                ax = axes[idx]
                ax.plot(k_values, A_values, 'o-', color=color, linewidth=2, markersize=4, alpha=0.7, label='A(k) computed')
                ax.axhline(params['A'], linestyle='--', color='black', linewidth=2, alpha=0.7, label=f"$p_0 \\cdot \\Gamma(\\alpha+\\gamma)/\\Gamma(\\alpha)$ = {params['A']:.2f}")
                ax.set_xlabel('k', fontsize=13)
                ax.set_ylabel('A(k)', fontsize=13)
                ax.set_title(f"Type '{node_type}' Normalization", fontsize=13, fontweight='bold')
                ax.set_xscale('log')
                ax.legend(fontsize=10, loc='best')
                ax.grid(True, alpha=0.3)
            legend_text = f"N={self.net.n0 + self.net.n_nodes:,}, m={self.net.m_edges}"
            fig.text(0.5, 0.98, legend_text, ha='center', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fm.save_fig(fig, "A_normalization")
            for node_type in ['a', 'b']:
                params = self.net._get_params(node_type)
                csv_data = [{'k': k, 'A_k': params['p0'] * (np.prod([(params['alpha'] + i) / (params['alpha'] + params['gamma'] + i) for i in range(k)]) if k > 0 else 1.0) * gamma_func(k + params['alpha'] + params['gamma']) / gamma_func(k + params['alpha']), 'A_asymptotic': params['A']} for k in k_values]
                df = pd.DataFrame(csv_data)
                filepath = fm.data_dir / f"A_normalization_type_{node_type}.csv"
                with open(filepath, 'w') as f:
                    f.write(f"# Normalization constant A(k)\n# Node type: {node_type}\n# p0: {params['p0']:.6f}, alpha: {params['alpha']:.6f}, gamma: {params['gamma']:.6f}\n# A_asymptotic: {params['A']:.6f}\n# Generated: {datetime.datetime.now().isoformat()}\n")
                    df.to_csv(f, index=False)
            return fig

        def _plot_normalization_power_law(self, fm: "FileManager", max_k: int, figsize: tuple):
                """
                For Sterling: show the analytic curve A*(k+alpha)^{-gamma}
                                together with the prefactor A as a horizontal line.
                For power_law_2: unchanged original behaviour.
                """
                fig, axes = plt.subplots(1, 2, figsize=figsize)

                for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):

                    if node_type not in self.net.power_law_params:
                        self.net._compute_power_law_params(node_type)

                    p      = self.net.power_law_params[node_type]
                    ax     = axes[idx]
                    k_vals = np.arange(0, max_k + 1)

                    if self.net.theory_type == 'Sterling':
                        A, alpha, gamma = p['A'], p['alpha'], p['gamma']
                        pmf_vals = A * (k_vals + alpha) ** (-gamma)

                        ax.plot(k_vals, pmf_vals, 'o-', color=color,
                                linewidth=2, markersize=4, alpha=0.7,
                                label=fr'$A\,(k+\alpha)^{{-\gamma}}$, '
                                    fr'$\alpha={alpha:.2f}$, $\gamma={gamma:.2f}$')
                        ax.axhline(A, linestyle='--', color='black', linewidth=2, alpha=0.7,
                                label=fr'$A = p_0\,\Gamma(\alpha+\gamma)/\Gamma(\alpha) = {A:.4f}$')

                        ax.set_xlabel('k', fontsize=13)
                        ax.set_ylabel('p(k)', fontsize=13)
                        ax.set_title(f"Type '{node_type}' Sterling's Approximation", fontsize=13, fontweight='bold')
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.legend(fontsize=9, loc='best')
                        ax.grid(True, alpha=0.3, which='both')

                        # CSV
                        csv_data = [{'k': int(k), 'pmf': A * (k + alpha) ** (-gamma),
                                    'A': A, 'alpha': alpha, 'gamma': gamma}
                                    for k in k_vals]
                        df = pd.DataFrame(csv_data)
                        filepath = fm.data_dir / f"Sterling_type_{node_type}.csv"
                        with open(filepath, 'w') as f:
                            f.write(f"# Sterling PMF: p(k) = A*(k+alpha)^(-gamma)\n"
                                    f"# Node type: {node_type}\n"
                                    f"# A: {A:.6f}, alpha: {alpha:.6f}, gamma: {gamma:.6f}\n"
                                    f"# Generated: {datetime.datetime.now().isoformat()}\n")
                            df.to_csv(f, index=False)

                    else:  # power_law_2 — original behaviour
                        alpha, k0, norm = p['alpha'], p.get('k0', 0.0), p['norm']
                        pmf_vals = ((k_vals + k0) ** (-alpha)) / norm

                        ax.plot(k_vals, pmf_vals, 'o-', color=color,
                                linewidth=2, markersize=4, alpha=0.7,
                                label=f'$\\alpha = {alpha:.2f}$'
                                    + (f', $k_0 = {k0:.2f}$' if k0 > 0 else ''))
                        ax.set_xlabel('k', fontsize=13)
                        ax.set_ylabel('p(k)', fontsize=13)
                        ax.set_title(f"Type '{node_type}' Power Law 2", fontsize=13, fontweight='bold')
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        ax.legend(fontsize=10, loc='best')
                        ax.grid(True, alpha=0.3, which='both')

                        csv_data = [{'k': int(k), 'pmf': ((k + k0) ** (-alpha)) / norm,
                                    'alpha': alpha, 'k0': k0, 'norm': norm}
                                    for k in k_vals]
                        df = pd.DataFrame(csv_data)
                        filepath = fm.data_dir / f"power_law_2_type_{node_type}.csv"
                        with open(filepath, 'w') as f:
                            f.write(f"# Power law 2 PMF: p(k) ~ (k+k0)^(-alpha)/norm\n"
                                    f"# Node type: {node_type}\n"
                                    f"# alpha: {alpha:.6f}, k0: {k0:.6f}, norm: {norm:.6f}\n"
                                    f"# Generated: {datetime.datetime.now().isoformat()}\n")
                            df.to_csv(f, index=False)

                legend_text = (f"N={self.net.n0 + self.net.n_nodes:,}, "
                            f"m={self.net.m_edges}, theory={self.net.theory_type}")
                fig.text(0.5, 0.98, legend_text, ha='center', va='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                fm.save_fig(fig, "power_law_params")
                return fig

class GoFDiagnostics:

    def theoretical_cdf_discrete(self, net: DirectedHomophilicNetwork, k, node_type: str, kmin: int, kmax: int, beta=None):
        """Theoretical CDF (dispatches by theory_type)."""
        k = np.atleast_1d(k)
        if net.theory_type == 'Diamond':
            if beta is None:
                beta = net.get_beta_for_type(node_type)
            return net.cdf_from_beta_truncated(k, beta, kmin, kmax)
        elif net.theory_type in ['Sterling', 'power_law_2']:
            return self._cdf_power_law_truncated(net, k, node_type, kmin, kmax)
        else:
            raise ValueError(f"Unknown theory_type: {net.theory_type}")
    
    def _cdf_power_law_truncated(self, net: DirectedHomophilicNetwork, k, node_type: str, kmin: int, kmax: int) -> np.ndarray:
        """Truncated CDF for power laws."""
        k = np.atleast_1d(k)
        k_int = np.floor(k).astype(int)
        support = np.arange(kmin, kmax + 1, dtype=int)
        pmf = net.theoretical_distribution(support, node_type)
        pmf = np.asarray(pmf, dtype=float)
        Z = pmf.sum()
        if Z <= 0:
            cdf_support = np.ones_like(support, dtype=float)
        else:
            pmf /= Z
            cdf_support = np.cumsum(pmf)
        
        F = np.zeros_like(k_int, dtype=float)
        s_min, s_max = support[0], support[-1]
        for i, ki in enumerate(k_int):
            if ki < s_min:
                F[i] = 0.0
            elif ki >= s_max:
                F[i] = 1.0
            else:
                F[i] = cdf_support[ki - s_min]
        return F.item() if F.size == 1 else F
    
    def _p_from_D_theory(self, d_e: float, N: int, j_max: int = 100) -> float:
        """
        Compute p(d_e, N) from the CSN-type approximation (Eq. 29 from paper).
        """
        if N <= 0 or d_e <= 0:
            return 0.0

        rootN = np.sqrt(N)
        term_base = d_e * rootN + 0.12 * d_e + 0.11 * d_e / rootN
        
        # Check if term_base is too large (would cause underflow)
        # exp(-700) ≈ 1e-304 is near machine precision limit
        threshold = np.sqrt(350)  # so that -2 * threshold^2 ≈ -700
        
        if term_base > threshold:
            # All exponential terms will underflow to 0, so p ≈ 0
            return 0.0
        
        arg_base = -2.0 * (term_base ** 2)
        
        j = np.arange(1, j_max + 1, dtype=float)
        signs = (-1.0) ** (j - 1.0)
        
        # Only compute for j where exp(arg_base * j^2) won't underflow
        # We want arg_base * j^2 > -700
        max_j_safe = int(np.sqrt(-700 / arg_base)) if arg_base < 0 else j_max
        max_j_safe = min(max_j_safe, j_max)
        
        if max_j_safe < 1:
            return 0.0
        
        j = j[:max_j_safe]
        signs = signs[:max_j_safe]
        exponents = np.exp(arg_base * (j ** 2))
        
        p_val = 2.0 * np.sum(signs * exponents)
        
        # Clamp to [0, 1] to handle numerical issues
        return float(np.clip(p_val, 0.0, 1.0))

    def csn_distance(self,net: DirectedHomophilicNetwork,data,a: int,b: int, beta, node_type: str,) -> float:
        """
        CSN/AD-style distance on a truncated window [a,b]:
            D(data; beta; [a,b]) = max_{n = a,...,b}| (N_n / N_a) - S(n; beta; [a,b]) |/ sqrt( S(n; beta; [a,b]) * (1 - S(n; beta; [a,b])) )
        """
        N_a = data.size
        n_vals = np.arange(a, b + 1, dtype=int)
        sorted_data = np.sort(data)

        # For each n, N_n = # { x_i >= n }
        idx_first_ge_n = np.searchsorted(sorted_data, n_vals, side='left')
        N_n = N_a - idx_first_ge_n

        # Empirical term (N_n / N_a)
        empirical_ratio = N_n / N_a

        # Theoretical conditional CDF S(n; beta; [a,b])
        S_n = self.theoretical_cdf_discrete(net, n_vals, node_type, kmin=a, kmax=b,beta=beta,)
        S_n = np.asarray(S_n, dtype=float)

        # Denominator sqrt(S (1-S)), with safeguard
        denom = np.sqrt(S_n * (1.0 - S_n))
        valid = denom > 0

        if not np.any(valid):
            return 0.0

        numer = np.abs(empirical_ratio[valid] - S_n[valid])
        D_vals = numer / denom[valid]
        D = float(np.max(D_vals))
        return D

    def _fit_beta_mle(self, net: DirectedHomophilicNetwork, data, a: int, b: int,
                    node_type: str, beta_init, mle_type: str = None):
        """
        MLE for beta = (p0, alpha, gamma) on truncated window [a, b].
        mle_type controls which PMF is used inside the likelihood.
        Defaults to net.theory_type if not specified.
        """
        mle_type = mle_type or net.theory_type

        data = np.asarray(data, dtype=int)
        params0 = net._get_params(node_type)

        if data.size == 0:
            return np.array([params0['p0'], params0['alpha'], params0['gamma']], dtype=float), True

        if beta_init is None:
            p0_0, alpha0, gamma0 = float(params0['p0']), float(params0['alpha']), float(params0['gamma'])
        else:
            p0_0, alpha0, gamma0 = np.asarray(beta_init, dtype=float)

        k_support = np.arange(a, b + 1, dtype=int)
        unique_k, counts = np.unique(data, return_counts=True)
        mask_window = (unique_k >= a) & (unique_k <= b)
        unique_k, counts = unique_k[mask_window], counts[mask_window]

        if counts.size == 0:
            return np.array([p0_0, alpha0, gamma0], dtype=float), True

        k_to_idx = {k: i for i, k in enumerate(k_support)}
        idxs = np.array([k_to_idx[k] for k in unique_k], dtype=int)

        zero_mask = (k_support == 0)
        pos_mask  = (k_support > 0)
        k_pos     = k_support[pos_mask]

        def _pmf_on_support(p0, alpha, gamma):
            """Evaluate the chosen PMF over k_support."""
            pmf = np.zeros_like(k_support, dtype=float)
            if mle_type == 'Diamond':
                if np.any(zero_mask):
                    pmf[zero_mask] = p0
                if np.any(pos_mask):
                    pmf[pos_mask] = p0 * np.exp(betaln(k_pos, alpha + gamma) - betaln(k_pos, alpha))
            elif mle_type == 'Sterling':
                A = p0 * gamma_func(alpha + gamma) / gamma_func(alpha)
                pmf = A * (k_support + alpha) ** (-gamma)
            elif mle_type == 'power_law_2':
                k0 = 1.0   # fixed shift as in original
                pmf = (k_support + k0) ** (-alpha)   # alpha plays exponent role; p0/gamma unused
            else:
                raise ValueError(f"Unknown mle_type: '{mle_type}'")
            return pmf

        def objective(theta):
            p0, alpha, gamma = theta
            pmf = _pmf_on_support(p0, alpha, gamma)
            Z = pmf.sum()
            if Z <= 0:
                return 1e10
            pmf_norm = np.clip(pmf / Z, 1e-300, 1.0)
            return -np.sum(counts * np.log(pmf_norm[idxs]))

        theta0  = np.array([p0_0, alpha0, gamma0], dtype=float)
        bounds  = [(1e-6, 1.0 - 1e-6), (1e-4, None), (1e-4, None)]
        res     = minimize(objective, theta0, method='L-BFGS-B', bounds=bounds)

        if not res.success:
            return np.array([p0_0, alpha0, gamma0], dtype=float), True
        return np.array(res.x, dtype=float), False

    def _build_b_grid(self,net: DirectedHomophilicNetwork,node_type: str,a: int,candidate_bs,b_min,b_max,n_b: int,b_grid_type: str,):
        if candidate_bs is not None:
            bs = np.array(candidate_bs, dtype=int)
            return bs

        degrees = np.array(net._get_degrees(node_type), dtype=int)
        if degrees.size == 0:
            return None  # caller handles this

        max_deg = int(degrees.max())

        b_min_eff = a + 1 if b_min is None else max(a + 1, b_min)
        b_max_eff = max_deg if b_max is None else min(max_deg, b_max)

        if b_min_eff > b_max_eff:
            return None

        if b_grid_type == 'linear':
            bs = np.linspace(b_min_eff, b_max_eff, n_b, dtype=int)
            bs = np.unique(bs)
        elif b_grid_type == 'log':
            bs = np.logspace(np.log10(b_min_eff), np.log10(b_max_eff), n_b, dtype=int)
            bs = np.unique(bs)
        else:
            raise ValueError(f"Unknown b_grid_type='{b_grid_type}'. Use 'linear' or 'log'.")

        return bs

    def mc_on_window(self, net: DirectedHomophilicNetwork, node_type: str, a: int, b: int,
                        beta_theory, N_sims: int, data_cache=None, mle_type: str = None):
            """
            Monte Carlo on window [a, b].
            mle_type controls which PMF is used in the MLE fitting step.
            Defaults to net.theory_type if not specified.
            """
            mle_type = mle_type or net.theory_type

            D_theory_list = []
            D_mle_list = []
            beta_mle_list = []
            fallback_flags = []

            beta_theory_arr = np.asarray(beta_theory) if beta_theory is not None else None

            if data_cache is None:
                print(f"  Generating {N_sims} network simulations for MC...")
                cache_list = []
                for sim_idx in range(N_sims):
                    net_sim = DirectedHomophilicNetwork(
                        n0=net.n0, n_nodes=net.n_nodes, m_edges=net.m_edges,
                        h=net.h, f_a=net.f_a, mu_a=net.mu['a'], mu_b=net.mu['b'], seed=None)
                    net_sim.generate_network()
                    degrees = np.asarray(net_sim._get_degrees(node_type), dtype=int)
                    cache_list.append(degrees)
            else:
                if isinstance(data_cache, np.ndarray):
                    cache_list = [np.asarray(data_cache, dtype=int)]
                elif isinstance(data_cache, list):
                    cache_list = [np.asarray(arr, dtype=int) for arr in data_cache]
                else:
                    raise TypeError(
                        f"data_cache must be None, np.ndarray, or list of arrays. "
                        f"Got {type(data_cache)}"
                    )
                actual_sims = len(cache_list)
                if actual_sims != N_sims:
                    print(
                        f"  ⚠️  data_cache has {actual_sims} samples but N_sims={N_sims}. "
                        f"Using {actual_sims} samples from cache."
                    )

            for sim_idx, degrees in enumerate(cache_list):
                data_s = degrees[(degrees >= a) & (degrees <= b)]

                if data_s.size == 0:
                    D_theory_list.append(0.0)
                    D_mle_list.append(0.0)
                    beta_mle_list.append(
                        beta_theory_arr if beta_theory_arr is not None
                        else np.array([np.nan, np.nan, np.nan])
                    )
                    fallback_flags.append(True)
                    continue

                D_theory_s = self.csn_distance(net, data_s, a, b, beta_theory, node_type)

                beta_mle_s, used_fallback_s = self._fit_beta_mle(
                    net, data_s, a, b, node_type, beta_init=None, mle_type=mle_type
                )
                D_mle_s = self.csn_distance(net, data_s, a, b, beta_mle_s, node_type)

                D_theory_list.append(D_theory_s)
                D_mle_list.append(D_mle_s)
                beta_mle_list.append(beta_mle_s)
                fallback_flags.append(used_fallback_s)

            D_theory_arr  = np.array(D_theory_list, dtype=float)
            D_mle_arr     = np.array(D_mle_list, dtype=float)
            fallback_flags = np.asarray(fallback_flags, dtype=bool)

            greater_mask = D_mle_arr > D_theory_arr
            p       = float(np.mean(greater_mask.astype(float)))
            sigma_p = float(np.sqrt(p * (1.0 - p) / max(len(D_theory_arr), 1)))

            try:
                beta_mle_arr = np.asarray(beta_mle_list, dtype=float)
            except (ValueError, TypeError):
                beta_mle_arr = np.array(beta_mle_list, dtype=object)

            frac_moved = float(np.mean(~fallback_flags)) if fallback_flags.size > 0 else np.nan

            beta_mle_mean = None
            beta_mle_std  = None
            if (isinstance(beta_mle_arr, np.ndarray)
                    and beta_mle_arr.ndim == 2
                    and beta_mle_arr.shape[0] == fallback_flags.size):
                mask_non_fallback = ~fallback_flags
                if np.any(mask_non_fallback):
                    beta_mle_non_fallback = beta_mle_arr[mask_non_fallback]
                    beta_mle_mean = np.nanmean(beta_mle_non_fallback, axis=0)
                    beta_mle_std  = np.nanstd(beta_mle_non_fallback, axis=0)

            percent_diff_mean_vs_theory = None
            if beta_theory_arr is not None and beta_mle_mean is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    percent_diff_mean_vs_theory = 100.0 * (
                        (beta_mle_mean - beta_theory_arr) / beta_theory_arr
                    )
                    percent_diff_mean_vs_theory = np.where(
                        beta_theory_arr == 0.0, np.nan, percent_diff_mean_vs_theory
                    )

            return {
                'a': a, 'b': b,
                'D_theory': D_theory_arr, 'D_mle': D_mle_arr,
                'beta_mle': beta_mle_arr,
                'p': p, 'sigma_p': sigma_p,
                'beta_mle_mean': beta_mle_mean, 'beta_mle_std': beta_mle_std,
                'beta_theory': beta_theory_arr,
                'frac_moved': frac_moved,
                'percent_diff_mean_vs_theory': percent_diff_mean_vs_theory,
                'mle_type': mle_type,
            }

    def scan_over_b(self, net: DirectedHomophilicNetwork, node_type: str, a: int,
                    beta_theory, candidate_bs=None, N_sims: int = 20, b_min: int = None,
                    b_max: int = None, n_b: int = 20, b_grid_type: str = 'linear',
                    data_cache: np.ndarray = None, mle_type: str = None):
        """
        Scan over b values. If data_cache is None, generate one set of N_sims
        non-truncated samples (degree arrays) and reuse across all b values.
        mle_type controls which PMF is used in the MLE fitting step.
        Defaults to net.theory_type if not specified.
        """
        mle_type = mle_type or net.theory_type

        bs = self._build_b_grid(net=net, node_type=node_type, a=a,
                                candidate_bs=candidate_bs, b_min=b_min,
                                b_max=b_max, n_b=n_b, b_grid_type=b_grid_type)

        if bs is None or len(bs) == 0:
            return {'a': a, 'beta_theory': beta_theory, 'windows': [], 'b_grid': None}

        if data_cache is None:
            data_cache_list = []
            for _ in range(N_sims):
                net_sim = DirectedHomophilicNetwork(
                    n0=net.n0, n_nodes=net.n_nodes, m_edges=net.m_edges,
                    h=net.h, f_a=net.f_a, mu_a=net.mu['a'], mu_b=net.mu['b'], seed=None)
                net_sim.generate_network()
                degrees = np.array(net_sim._get_degrees(node_type), dtype=int)
                data_cache_list.append(degrees)
            data_cache = data_cache_list
        else:
            if not isinstance(data_cache, list):
                data_cache = [np.asarray(data_cache, dtype=int)]

        windows_results = []
        for b in bs:
            if b <= a:
                raise ValueError(f"Each b_j must satisfy b_j > a. Got a={a}, b_j={b}.")

            res_b = self.mc_on_window(
                net=net, node_type=node_type, a=a, b=int(b),
                beta_theory=beta_theory, N_sims=len(data_cache),
                data_cache=data_cache, mle_type=mle_type
            )

            windows_results.append({
                'a': a, 'b': int(b),
                'D_theory': res_b['D_theory'], 'D_mle': res_b['D_mle'],
                'beta_mle': res_b['beta_mle'],
                'p': res_b['p'], 'sigma_p': res_b['sigma_p'],
                'beta_mle_mean': res_b['beta_mle_mean'],
                'beta_mle_std':  res_b['beta_mle_std'],
                'beta_theory':   res_b['beta_theory'],
                'frac_moved':    res_b['frac_moved'],
                'percent_diff_mean_vs_theory': res_b['percent_diff_mean_vs_theory'],
                'mle_type': mle_type,
            })

        return {
            'a': a,
            'beta_theory': beta_theory,
            'windows': windows_results,
            'b_grid': np.array(bs, dtype=int),
            'mle_type': mle_type,
        }

    def scan_until_threshold(self, windows, a: int,p_c: float,z: float = 1.0,):
        """
        Find largest b such that p(b) - z * sigma_p(b) >= p_c i.e. lower confidence bound exceeds threshold.
        """
        if not windows:
            return {
                'a': a,
                'p_c': float(p_c),
                'windows_evaluated': [],
                'largest_window': None,
            }

        windows_sorted = sorted(windows, key=lambda w: w['b'])

        windows_evaluated = []
        largest_window_info = None

        for w in reversed(windows_sorted):
            b = int(w['b'])
            p_val = float(w['p'])
            sigma_p = float(w['sigma_p'])

            lower_bound = p_val - z * sigma_p
            windows_evaluated.append(w)

            if lower_bound >= p_c:
                largest_window_info = {
                    'a': a,
                    'b': b,
                    'p': p_val,
                    'sigma_p': sigma_p,
                    'index': 0,
                }
                break

        return {
            'a': a,
            'p_c': float(p_c),
            'windows_evaluated': windows_evaluated,
            'largest_window': largest_window_info,
        }

    def select_largest_window_for_pcs(self, windows, a: int, p_c_list,):
        """
        For a given list of windows (from scan_over_b['windows']) and
        a list of p_c values, return a dict: p_c_value: {scan_until_threshold_result_for_that_p_c,...}
        """
        results = {}
        for p_c in p_c_list:
            res = self.scan_until_threshold(windows=windows, a=a, p_c=p_c)
            results[p_c] = res
        return results

    def _csn_sweep_core(self, base_net: DirectedHomophilicNetwork, sweep_param_values: np.ndarray,
                            vary: str, node_type: str, a: int, candidate_bs, b_min: int, b_max: int,
                            n_b: int, b_grid_type: str, N_sims: int, p_c_list, figsize=(12, 6),
                            mle_type: str = None):

            mle_type = mle_type or base_net.theory_type
            plt.rcParams['font.family'] = 'Times New Roman'

            windows_info    = []
            frac_nodes_kept = {p_c: [] for p_c in p_c_list}
            frac_edges_kept = {p_c: [] for p_c in p_c_list}
            no_window       = {p_c: [] for p_c in p_c_list}

            label = 'm_edges' if vary == 'm_edges' else 'n0'
            print(f"\nCSN sweep over {label}: {sweep_param_values}, mle_type={mle_type}")

            for val in sweep_param_values:
                if vary == 'm_edges':
                    n0, n_nodes, m_edges = base_net.n0, base_net.n_nodes, int(val)
                elif vary == 'n0':
                    n0, n_nodes, m_edges = int(val), base_net.n_nodes, base_net.m_edges
                else:
                    raise ValueError("vary must be 'm_edges' or 'n0'")

                print(f"\n  {label} = {val}: generating network and scanning windows...")

                net_temp = DirectedHomophilicNetwork(
                    n0=n0, n_nodes=n_nodes, m_edges=m_edges,
                    h=base_net.h, f_a=base_net.f_a,
                    mu_a=base_net.mu['a'], mu_b=base_net.mu['b'])
                net_temp.generate_network()

                # Bootstrap so _get_params() is always populated
                net_temp.theory_type = 'Diamond'
                net_temp._fit_asymptotes()
                net_temp._compute_power_law_params('a')
                net_temp._compute_power_law_params('b')
                net_temp.theory_type = base_net.theory_type   # restore

                degrees      = np.array(net_temp._get_degrees(node_type), dtype=int)
                beta_theory  = net_temp.get_beta_for_type(node_type)

                scan_res = self.scan_over_b(
                    net=net_temp, node_type=node_type, a=a, beta_theory=beta_theory,
                    candidate_bs=candidate_bs, N_sims=N_sims,
                    b_min=b_min, b_max=b_max, n_b=n_b, b_grid_type=b_grid_type,
                    mle_type=mle_type)

                scan_res['degrees_for_node_type'] = degrees.copy()
                windows_info.append(scan_res)

                windows = scan_res['windows']
                bs      = scan_res['b_grid']

                if bs is None or len(bs) == 0 or not windows:
                    print("    No valid b range; skipping.")
                    for p_c in p_c_list:
                        frac_nodes_kept[p_c].append(0.0)
                        frac_edges_kept[p_c].append(0.0)
                        no_window[p_c].append(True)
                    continue

                grid_str = (f"[{bs[0]}, {bs[1]},... {bs[-2]}, {bs[-1]}]"
                            if len(bs) > 6 else str(bs.tolist()))
                print(f"    b-grid (candidate b_j): {grid_str}")

                full_w = max(windows, key=lambda w: w['b'])
                print(f"    no-trunc p={full_w['p']:.4f} ± {full_w['sigma_p']:.4f}")

                total_edges = degrees.sum() if degrees.size > 0 else 0
                pcs_results = self.select_largest_window_for_pcs(
                    windows=windows, a=a, p_c_list=p_c_list)

                for p_c in p_c_list:
                    lw = pcs_results[p_c]['largest_window']
                    if lw is None:
                        frac_nodes_kept[p_c].append(0.0)
                        frac_edges_kept[p_c].append(0.0)
                        no_window[p_c].append(True)
                        print(f"    p_c={p_c}: no acceptable window")
                        continue

                    b_star  = lw['b']
                    p_val   = lw['p']
                    sigma_p = lw['sigma_p']

                    mask_keep = (degrees >= a) & (degrees <= b_star)
                    fn = mask_keep.mean()
                    fe = degrees[mask_keep].sum() / total_edges if total_edges > 0 else 0.0

                    frac_nodes_kept[p_c].append(fn)
                    frac_edges_kept[p_c].append(fe)
                    no_window[p_c].append(False)

                    print(f"    p_c={p_c}: b*={b_star}, "
                        f"p-z*σ={p_val - sigma_p:.4f}, "
                        f"nodes={fn:.4f}, edges={fe:.4f}")

            # ── plots ─────────────────────────────────────────────────────────────
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            x_values = sweep_param_values
            x_label  = 'm_edges' if vary == 'm_edges' else 'n0'
            frac_new_nodes = (base_net.n_nodes / (base_net.n_nodes + base_net.n0)
                            if vary == 'm_edges'
                            else base_net.n_nodes / (base_net.n_nodes + sweep_param_values))

            colors = plt.cm.viridis(np.linspace(0, 1, len(p_c_list)))

            for color, p_c in zip(colors, p_c_list):
                fn_arr = np.array(frac_nodes_kept[p_c], dtype=float)
                nw_arr = np.array(no_window[p_c], dtype=bool)
                ax1.plot(x_values[~nw_arr], fn_arr[~nw_arr],
                        marker='o', linestyle='-', linewidth=2, markersize=5,
                        color=color, label=f'p_c = {p_c}')
                ax1.scatter(x_values[nw_arr], np.zeros_like(fn_arr[nw_arr]),
                            marker='x', s=60, color=color, zorder=3)

            if np.isscalar(frac_new_nodes):
                ax1.axhline(frac_new_nodes, linestyle='--', color='black', linewidth=1.5)

            ax1.set_xlabel(x_label)
            ax1.set_ylabel('Fraction of nodes kept')
            ax1.set_ylim(-0.05, 1.05)
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            for color, p_c in zip(colors, p_c_list):
                fe_arr = np.array(frac_edges_kept[p_c], dtype=float)
                nw_arr = np.array(no_window[p_c], dtype=bool)
                ax2.plot(x_values[~nw_arr], fe_arr[~nw_arr],
                        marker='o', linestyle='-', linewidth=2, markersize=5,
                        color=color, label=f'p_c = {p_c}')
                ax2.scatter(x_values[nw_arr], np.zeros_like(fe_arr[nw_arr]),
                            marker='x', s=50, color=color, zorder=3)

            ax2.set_xlabel(x_label)
            ax2.set_ylabel('Fraction of edges kept')
            ax2.set_ylim(-0.05, 1.05)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()

            return {
                'x_values': x_values,
                'frac_nodes_kept': frac_nodes_kept,
                'frac_edges_kept': frac_edges_kept,
                'no_window': no_window,
                'windows_info': windows_info,
                'fig': fig,
                'vary': vary,
                'mle_type': mle_type,
            }

    def csn_sweep_m_edges(self, net: DirectedHomophilicNetwork, m_min: int, m_max: int,
                        m_step: int = 1, node_type: str = 'b', a: int = 0,
                        candidate_bs=None, b_min: int = None, b_max: int = None,
                        n_b: int = 20, b_grid_type: str = 'linear', N_sims: int = 20,
                        p_c_list=(0.1, 0.2, 0.4, 0.6), figsize=(12, 6),
                        mle_type: str = None):
        m_values = np.arange(m_min, m_max + 1, m_step, dtype=int)
        res = self._csn_sweep_core(
            base_net=net, sweep_param_values=m_values, vary='m_edges',
            node_type=node_type, a=a, candidate_bs=candidate_bs,
            b_min=b_min, b_max=b_max, n_b=n_b, b_grid_type=b_grid_type,
            N_sims=N_sims, p_c_list=p_c_list, figsize=figsize, mle_type=mle_type)
        return {
            'm_values':        res['x_values'],
            'frac_nodes_kept': res['frac_nodes_kept'],
            'frac_edges_kept': res['frac_edges_kept'],
            'no_window':       res['no_window'],
            'windows_info':    res['windows_info'],
            'fig':             res['fig'],
            'mle_type':        res['mle_type'],
        }

    def csn_sweep_n0(self, net: DirectedHomophilicNetwork, n0_min: int, n0_max: int,
                    n0_step: int = 1, node_type: str = 'b', a: int = 0,
                    candidate_bs=None, b_min: int = None, b_max: int = None,
                    n_b: int = 20, b_grid_type: str = 'linear', N_sims: int = 20,
                    p_c_list=(0.1, 0.2, 0.4, 0.6), figsize=(12, 6),
                    mle_type: str = None):
        n0_values = np.arange(n0_min, n0_max + 1, n0_step, dtype=int)
        res = self._csn_sweep_core(
            base_net=net, sweep_param_values=n0_values, vary='n0',
            node_type=node_type, a=a, candidate_bs=candidate_bs,
            b_min=b_min, b_max=b_max, n_b=n_b, b_grid_type=b_grid_type,
            N_sims=N_sims, p_c_list=p_c_list, figsize=figsize, mle_type=mle_type)
        return {
            'n0_values':       res['x_values'],
            'frac_nodes_kept': res['frac_nodes_kept'],
            'frac_edges_kept': res['frac_edges_kept'],
            'no_window':       res['no_window'],
            'windows_info':    res['windows_info'],
            'fig':             res['fig'],
            'mle_type':        res['mle_type'],
        }

    def _select_indices_for_diagnostics(self, n: int, k: int):

        if k <= 0 or n <= 0:
            return []
        if k >= n:
            return list(range(n))

        positions = np.linspace(0, n - 1, k)
        indices = sorted(set(int(round(p)) for p in positions))
        return indices

    def _p_from_D_theory_over_b(self, scan_res, net: DirectedHomophilicNetwork, node_type: str,):
        """
        For each window in scan_res['windows'], compute p(d_e, N)
        where d_e is D_theory for that window and N is the effective
        sample size (number of degrees in [a, b]) for that truncation.
        """
        windows = scan_res.get('windows', [])
        if not windows:
            return np.array([]), np.array([])

        a = scan_res.get('a', 0)

        degrees = np.array(net._get_degrees(node_type), dtype=int)
        if degrees.size == 0:
            return np.array([]), np.array([])

        bs = []
        p_vals = []

        for w in windows:
            b = int(w['b'])
            D_theory_arr = np.asarray(w['D_theory'], dtype=float)
            if D_theory_arr.size == 0:
                continue

            # Use the mean D_theory across MC simulations for this window
            d_e = float(np.mean(D_theory_arr))

            mask = (degrees >= a) & (degrees <= b)
            N_eff = int(np.sum(mask))

            p_de = self._p_from_D_theory(d_e=d_e, N=N_eff)
            bs.append(b)
            p_vals.append(p_de)

        return np.asarray(bs, dtype=int), np.asarray(p_vals, dtype=float)

    def csn_sweep_2d_grid(self, net: DirectedHomophilicNetwork, m_min: int, m_max: int,
                            m_step: int, n0_min: int, n0_max: int, n0_step: int,
                            node_type: str = 'b', a: int = 0, candidate_bs=None,
                            b_min: int = None, b_max: int = None, n_b: int = 20,
                            b_grid_type: str = 'linear', N_sims: int = 20,
                            p_c_list=(0.1, 0.2, 0.4, 0.6), z: float = 1.0,
                            mle_type: str = None):
            """
            2D sweep over m_edges and n0. Skips invalid region where n0 <= m.
            mle_type controls which PMF is used in the MLE fitting step.
            Defaults to net.theory_type if not specified.
            """
            mle_type = mle_type or net.theory_type

            m_values  = np.arange(m_min,  m_max  + 1, m_step,  dtype=int)
            n0_values = np.arange(n0_min, n0_max + 1, n0_step, dtype=int)

            frac_nodes_grid = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
            frac_edges_grid = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
            b_star_grid     = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
            p_value_grid    = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}

            print(f"\n2D Grid: m ∈ [{m_min}, {m_max}], n0 ∈ [{n0_min}, {n0_max}], mle_type={mle_type}")

            total_points = len(n0_values) * len(m_values)
            valid_points = 0

            for i, n0 in enumerate(n0_values):
                for j, m in enumerate(m_values):
                    if n0 <= m:
                        continue

                    valid_points += 1
                    print(f"\n  Point {valid_points}/{total_points}: (m={m}, n0={n0})")

                    net_temp = DirectedHomophilicNetwork(
                        n0=n0, n_nodes=net.n_nodes, m_edges=m,
                        h=net.h, f_a=net.f_a, mu_a=net.mu['a'], mu_b=net.mu['b'])
                    net_temp.generate_network()

                    # Bootstrap Diamond params so _get_params() works for any theory
                    net_temp.theory_type = 'Diamond'
                    net_temp._fit_asymptotes()
                    net_temp._compute_power_law_params('a')
                    net_temp._compute_power_law_params('b')
                    net_temp.theory_type = net.theory_type   # restore to caller's theory

                    degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                    if degrees.size == 0:
                        continue

                    beta_theory = net_temp.get_beta_for_type(node_type)
                    scan_res = self.scan_over_b(
                        net=net_temp, node_type=node_type, a=a,
                        beta_theory=beta_theory, candidate_bs=candidate_bs,
                        N_sims=N_sims, b_min=b_min, b_max=b_max,
                        n_b=n_b, b_grid_type=b_grid_type, mle_type=mle_type
                    )

                    windows = scan_res['windows']
                    if not windows:
                        continue

                    full_w = max(windows, key=lambda w: w['b'])
                    print(f"    no-trunc p={full_w['p']:.4f} ± {full_w['sigma_p']:.4f}")

                    pcs_results = self.select_largest_window_for_pcs(
                        windows=windows, a=a, p_c_list=p_c_list)
                    total_edges = degrees.sum() if degrees.size > 0 else 0

                    for p_c in p_c_list:
                        lw = pcs_results[p_c]['largest_window']
                        if lw is None:
                            frac_nodes_grid[p_c][i, j] = 0.0
                            frac_edges_grid[p_c][i, j] = 0.0
                            continue

                        b_star      = lw['b']
                        p_val       = lw['p']
                        sigma_p     = lw['sigma_p']
                        lower_bound = p_val - z * sigma_p

                        mask_keep = (degrees >= a) & (degrees <= b_star)
                        fn = mask_keep.mean()
                        fe = degrees[mask_keep].sum() / total_edges if total_edges > 0 else 0.0

                        frac_nodes_grid[p_c][i, j] = fn
                        frac_edges_grid[p_c][i, j] = fe
                        b_star_grid[p_c][i, j]     = b_star
                        p_value_grid[p_c][i, j]    = lower_bound

                        print(f"    p_c={p_c}: b*={b_star}, p-z*σ={lower_bound:.4f}, "
                            f"nodes={fn:.4f}, edges={fe:.4f}")

            return {
                'm_values': m_values, 'n0_values': n0_values,
                'frac_nodes_grid': frac_nodes_grid, 'frac_edges_grid': frac_edges_grid,
                'b_star_grid': b_star_grid, 'p_value_grid': p_value_grid,
                'p_c_list': p_c_list, 'z': z, 'a': a, 'node_type': node_type,
                'mle_type': mle_type,
            }
    class CSN1DVis:
        """1D sweep visualization for m_edges and n0 sweeps."""
        
        def __init__(self, parent_gof: "GoFDiagnostics"):
            self.gof = parent_gof
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['axes.linewidth'] = 0.8
            plt.rcParams['xtick.major.width'] = 0.8
            plt.rcParams['ytick.major.width'] = 0.8
        
        def plot_frac_sweep(self, x_values, frac_nodes_dict, frac_edges_dict, 
                            no_window_dict, p_c_list, x_name, node_type, a, 
                            metric='nodes', figsize=(10, 6)):
            """Plot fraction (nodes or edges) vs sweep parameter.
            
            Args:
                metric: 'nodes' or 'edges'
            """
            frac_dict = frac_nodes_dict if metric == 'nodes' else frac_edges_dict
            
            fig, ax = plt.subplots(figsize=figsize)
            colors = plt.cm.viridis(np.linspace(0, 1, len(p_c_list)))
            
            for color, p_c in zip(colors, p_c_list):
                frac_arr = np.array(frac_dict[p_c], dtype=float)
                nw_arr = np.array(no_window_dict[p_c], dtype=bool)
                
                ok_mask = ~nw_arr
                fail_mask = nw_arr
                
                ax.plot(x_values[ok_mask], frac_arr[ok_mask],
                    marker='o', linestyle='-', linewidth=2, markersize=5,
                    color=color, label=f'$p_c = {p_c}$')
                
                ax.scatter(x_values[fail_mask], np.zeros_like(frac_arr[fail_mask]),
                        marker='x', s=60, color=color, zorder=3)
            
            legend_text = f"{x_name} = sweep, node_type = {node_type}, a = {a}"
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(x_name, fontsize=12)
            ax.set_ylabel(f'Fraction of {metric} kept', fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='lower right')
            plt.tight_layout()
            return fig
        
        def plot_beta_mle_pct_diff(self, windows, x_val, x_name, node_type, a, figsize=(10, 6)):
            """Plot % difference between fitted and theory beta across truncations."""
            bs = []
            p0_pct = []
            p0_err = []
            alpha_pct = []
            alpha_err = []
            gamma_pct = []
            gamma_err = []
            
            for w in windows:
                b = int(w['b'])
                beta_mle_mean = w.get('beta_mle_mean')
                beta_mle_std = w.get('beta_mle_std')
                beta_theory = w.get('beta_theory')
                
                if beta_mle_mean is None or beta_theory is None:
                    continue
                
                p0_m, alpha_m, gamma_m = beta_mle_mean
                p0_s, alpha_s, gamma_s = beta_mle_std if beta_mle_std is not None else (0, 0, 0)
                p0_t, alpha_t, gamma_t = beta_theory
                
                bs.append(b)
                
                p0_pct.append(100 * (p0_m - p0_t) / p0_t if p0_t != 0 else np.nan)
                p0_err.append(100 * p0_s / p0_t if p0_t != 0 else np.nan)
                
                alpha_pct.append(100 * (alpha_m - alpha_t) / alpha_t if alpha_t != 0 else np.nan)
                alpha_err.append(100 * alpha_s / alpha_t if alpha_t != 0 else np.nan)
                
                gamma_pct.append(100 * (gamma_m - gamma_t) / gamma_t if gamma_t != 0 else np.nan)
                gamma_err.append(100 * gamma_s / gamma_t if gamma_t != 0 else np.nan)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.errorbar(bs, p0_pct, yerr=p0_err, fmt='o-', linewidth=2, markersize=5,
                    color='red', label=r'$p_0$', capsize=4)
            ax.errorbar(bs, alpha_pct, yerr=alpha_err, fmt='s-', linewidth=2, markersize=5,
                    color='blue', label=r'$\alpha$', capsize=4)
            ax.errorbar(bs, gamma_pct, yerr=gamma_err, fmt='^-', linewidth=2, markersize=5,
                    color='green', label=r'$\gamma$', capsize=4)
            
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            legend_text = f"{x_name} = {x_val}, node_type = {node_type}, a = {a}"
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(r'$b$ (truncation upper bound)', fontsize=12)
            ax.set_ylabel(r'% difference from theory', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11, loc='best')
            plt.tight_layout()
            return fig

        def plot_p_vs_b(self, scan_res, p_c_list, x_val, x_name, node_type, a, z=1.0, figsize=(10, 6)):
            """Plot p(b) vs b with error bars and lower bounds."""
            windows = scan_res.get('windows', [])
            if not windows:
                return None
            
            bs      = np.array([w['b'] for w in windows], dtype=int)
            ps      = np.array([w['p'] for w in windows], dtype=float)
            sigmas  = np.array([w['sigma_p'] for w in windows], dtype=float)
            p_lower = ps - z * sigmas
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.errorbar(bs, ps, yerr=sigmas, fmt='none', ecolor='0.6',
                    elinewidth=1.5, capsize=0, zorder=1)
            
            for b, p_val, s in zip(bs, ps, sigmas):
                for y in (p_val + s, p_val, p_val - s):
                    ax.plot([b - 0.1, b + 0.1], [y, y], color='orange',
                        linewidth=1.5, zorder=2)
            
            ax.plot(bs, p_lower, '-s', color='C1', linewidth=2, markersize=5,
                label=fr'Lower bound $p(b) - {z}\,\sigma_p$', zorder=3)
            
            colors_pc = plt.cm.plasma(np.linspace(0, 1, len(p_c_list)))
            for color_pc, p_c in zip(colors_pc, p_c_list):
                ax.axhline(y=p_c, color=color_pc, linestyle='--', linewidth=1.5,
                        label=fr'$p_c = {p_c}$')
            
            legend_text = f"{x_name} = {x_val}, node_type = {node_type}, a = {a}"
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(r'$b$ (upper truncation)', fontsize=12)
            ax.set_ylabel(r'$p(b)$ and lower bounds', fontsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9)
            plt.tight_layout()
            
            # CSV export
            csv_data = {
                'b': bs,
                'p': ps,
                'sigma_p': sigmas,
                'p_lower': p_lower,
            }
            for p_c in p_c_list:
                csv_data[f'p_c_{p_c}'] = np.full_like(bs, p_c, dtype=float)
            
            df = pd.DataFrame(csv_data)
            # Use self.gof to access parent for file manager access
            # Actually we need to return the dataframe and let __main__ handle export
            # Store it in the figure's metadata for later retrieval
            fig._csv_data = df
            fig._csv_metadata = {
                'x_name': x_name,
                'x_val': x_val,
                'node_type': node_type,
                'a': a,
                'z': z,
            }
            
            return fig 
        
    class CSN2DVis:
        """
        2D visualization utilities for CSN-based windows.
        """
        def __init__(self, parent_gof: "GoFDiagnostics"):
            self.gof = parent_gof
            
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['TeX Gyre Termes', 'Times New Roman']
            plt.rcParams['mathtext.fontset'] = 'cm'
            plt.rcParams['axes.linewidth'] = 0.8
            plt.rcParams['xtick.major.width'] = 0.8
            plt.rcParams['ytick.major.width'] = 0.8

        def _create_csn_colormap(self):
            """Custom colormap for CSN plot."""
            viridis = plt.colormaps['viridis'] 
            viridis_colors = viridis(np.linspace(0, 1, 256))
            
            custom_colors = [
                (0.85, 0.85, 0.85, 1.0),  # Light gray: Not assessed
                (0.05, 0.05, 0.25, 1.0),  # Dark navy: No valid range
            ]
            custom_colors.extend(viridis_colors)
            
            return ListedColormap(custom_colors)

        def _prepare_data_for_csn_colormap(self, data_grid, valid_mask):
            """Prepare data for CSN colormap with semantic encoding."""
            data_prepared = data_grid.copy()
            data_prepared[~valid_mask] = -0.5
            return data_prepared

        def _add_boundary_lines(self, ax, m_values, n0_values, valid_mask):
            """Add boundary lines between invalid and valid regions."""
            m_boundary = np.linspace(m_values.min(), m_values.max(), 100)
            ax.plot(m_boundary, m_boundary, 'k-', linewidth=1.5, alpha=0.7, zorder=10)
            ax.fill_between(m_values, m_values, n0_values[0], 
                        color='gray', alpha=0.25, zorder=1, label='Not assessed')
            
        def plot_contour_generic(self, m_values, n0_values, data_grid, 
                                metric_label, use_csn_cmap=False, figsize=(6, 5)):
            """Plot 2D contour from grid data.
            
            Args:
                use_csn_cmap: True for [0,1] metrics (frac_nodes, frac_edges), 
                            False for diverging metrics (margin)
            """
            M, N0 = np.meshgrid(m_values, n0_values)
            valid_mask = N0 > M
            data = data_grid.copy()
            data[~valid_mask] = np.nan
            
            fig, ax = plt.subplots(figsize=figsize)
            
            if use_csn_cmap:
                cmap = self._create_csn_colormap()
                data_prep = self._prepare_data_for_csn_colormap(data, valid_mask)
                norm = Normalize(vmin=-0.5, vmax=1.0)
                cs = ax.contourf(M, N0, data_prep, levels=20, cmap=cmap, norm=norm, extend='neither')
                cbar = fig.colorbar(cs, ax=ax, extend='neither', 
                                ticks=[-0.5, 0, 0.25, 0.5, 0.75, 1.0])
                cbar.ax.set_yticklabels(['', '0', '0.25', '0.5', '0.75', '1.0'])
                
                legend_elements = [
                    mpatches.Patch(facecolor=(0.85, 0.85, 0.85), edgecolor='black', 
                                linewidth=0.8, label='Not assessed'),
                    mpatches.Patch(facecolor=(0.05, 0.05, 0.25), edgecolor='black', 
                                linewidth=0.8, label='No valid range'),
                    mpatches.Patch(facecolor=(0.267, 0.004, 0.329), edgecolor='black', 
                                linewidth=0.8, label=r'Valid ($p \ll 1$)'),
                    mpatches.Patch(facecolor=(1.0, 1.0, 0.0), edgecolor='black', 
                                linewidth=0.8, label=r'Valid ($p \approx 1$)'),
                ]
                ax.legend(handles=legend_elements, loc='upper left', fontsize=8.5,
                        framealpha=0.95, edgecolor='black', fancybox=False)
            else:
                max_abs = np.nanmax(np.abs(data))
                max_abs = 1.0 if not np.isfinite(max_abs) or max_abs == 0 else max_abs
                levels = np.linspace(-max_abs, max_abs, 21)
                cs = ax.contourf(M, N0, data, levels=levels, cmap='coolwarm', extend='neither')
                cbar = fig.colorbar(cs, ax=ax, extend='neither')
                self._add_boundary_lines(ax, m_values, n0_values, valid_mask)
            
            cbar.set_label(metric_label, fontsize=11)
            ax.set_xlabel(r'$m$', fontsize=12)
            ax.set_ylabel(r'$n_0$', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            return fig                    

class NetworkStatistics:
    """
    Statistical reporting utilities for DirectedHomophilicNetwork.
    Methods take a `DirectedHomophilicNetwork` instance `net`.
        """
    def print_statistics(self, net: DirectedHomophilicNetwork):
        """Print comprehensive network statistics."""
        in_degrees_a = np.asarray(net._get_degrees('a'), dtype=int)
        in_degrees_b = np.asarray(net._get_degrees('b'), dtype=int)

        # Z-factor analysis
        if net.Z_factor:
            type_val = net.node_types[: net.graph.number_of_nodes()]
            in_deg = np.array([net.graph.in_degree(n) for n in net.graph.nodes()])

            lambda_vals = np.where(type_val == 0, net.lambda_a, net.lambda_b)
            mu_vals = np.where(type_val == 0, net.mu['a'], net.mu['b'])

            Z_emp = np.sum(lambda_vals * in_deg + mu_vals)
            Z_theo = net.graph.number_of_nodes() * net.Z_factor
            ratio = Z_emp / Z_theo

            print(f"\nZ-factor Analysis:")
            print(f"  Z_empirical/Z_theoretical = {ratio:.6f}")
            print(f"  % difference = {(ratio - 1.0) * 100:+.4f}%")

        # g_b comparison
        if net.g_b_empirical is not None:
            ratio_gb = net.g_b / net.g_b_empirical
            print(f"\ng_b Analysis:")
            print(f"  g_b (m - g_a)/g_b_empirical = {ratio_gb:.6f}")
            print(f"  % difference = {(ratio_gb - 1.0) * 100:+.4f}%")

        # Type-specific statistics
        for degrees, type_name in [(in_degrees_a, 'a'), (in_degrees_b, 'b')]:
            n_total = net.graph.number_of_nodes()
            n_type = len(degrees)
            
            print(f"\nType '{type_name}': {n_type:,} nodes ({n_type/n_total*100:.1f}%)")
            
            if degrees.size > 0:
                print(f"  Mean in-degree: {np.mean(degrees):.2f}")
                print(f"  Max in-degree: {np.max(degrees)}")
                print(f"  Min in-degree: {np.min(degrees)}")
            else:
                print(f"  Mean in-degree: N/A")
                print(f"  Max in-degree: N/A")
                print(f"  Min in-degree: N/A")

        print(
            f"\n_g_a_ = {net.g_a:.6f}, _g_b_ = {net.g_b:.6f}, "
            f"g_b_empirical = {net.g_b_empirical:.6f}"
        )
        print(
            f"g_a + g_b = {net.g_a + net.g_b:.6f} "
            f"(m = {net.m_edges})"
        )

config = dict(
    theory=dict(
    network_basic = ['Diamond', 'Sterling',], gof = 'Diamond', mle = 'Diamond',),
    plots=dict(network_basic= True, sweep_m_edges_csn=False, sweep_n0_csn=False,
               csn_p_vs_b_m=5, csn_p_vs_b_n0=5, grid_2d_sweep=False),
    network=dict(n0=4, n_nodes=28000, m_edges=3, h=0.7, f_a=0.55, mu_a=1, mu_b=1,
                 seed=5, power_law_params=None),
    sweep_m=dict(m_min=1, m_max=34, m_step=5, node_type='b', a=0,
                 p_c_list=[0.1, 0.2, 0.4], N_sims=30,
                 b_grid_type='linear', n_b=5, b_min=None, b_max=None),
    sweep_n0=dict(n0_min=10, n0_max=150, n0_step=30, node_type='b', a=0,
                  p_c_list=[0.1, 0.2, 0.4], N_sims=30,
                  b_grid_type='linear', n_b=5, b_min=None, b_max=None),
    grid_2d=dict(m_min=1, m_max=11, m_step=2, n0_min=2, n0_max=10, n0_step=2,
                 node_type='b', a=0,
                 p_c_list=[0.1, 0.2, 0.4], N_sims=50,
                 b_grid_type='linear', n_b=50, b_min=None, b_max=None, z=1.0))

if __name__ == "__main__":

    fm    = FileManager(config)
    gof   = GoFDiagnostics()
    stats = NetworkStatistics()

    theory_cfg = config["theory"]
    gof_theory = theory_cfg["gof"]
    mle_theory = theory_cfg["mle"]

    # theory_type on the network drives CDF dispatch in GoF
    net = DirectedHomophilicNetwork(**config["network"], theory_type=gof_theory)
    start = time.time()
    net.generate_network()
    print(f"Network generated in {time.time() - start:.2f}s")

    # Bootstrap: always needed so _get_params() is populated
    net.theory_type = 'Diamond'
    net._fit_asymptotes()
    net._compute_power_law_params('a')
    net._compute_power_law_params('b')
    net.theory_type = gof_theory   # restore

    stats.print_statistics(net)

    plots = config["plots"]

    # --- Network basic plots: one curve per theory in network_basic list ---
    if plots["network_basic"]:
        vis = DirectedHomophilicNetwork.NetworkVis(net)
        plot_network = True
        if plot_network:
            vis.plot_network_graph(fm) 
        vis.plot_degree_distributions_log(fm,   theories=theory_cfg["network_basic"])
        vis.plot_degree_distributions_linear(fm, theories=theory_cfg["network_basic"])

        for theory in theory_cfg["network_basic"]:
            print(f"Fitting {theory}")
            net.theory_type = theory
            if theory == 'Diamond':
                net._fit_asymptotes()
            else:
                net._compute_power_law_params('a')
                net._compute_power_law_params('b')
            vis = DirectedHomophilicNetwork.NetworkVis(net)
            vis.plot_asymptotes(fm)
            vis.plot_normalization(fm)

        net.theory_type = gof_theory   # restore after per-theory loop

    # --- GoF sweeps: use gof_theory + mle_theory from config['theory'] ---
    print(f"GoF analysis: gof={gof_theory}, mle={mle_theory}")

    if plots["sweep_m_edges_csn"]:
        results_m = gof.csn_sweep_m_edges(net, **config["sweep_m"],
                                           mle_type=mle_theory)
        fm.save_fig_with_suffix(results_m["fig"], "sweep_m_edges_csn", gof_theory)
        frac_data = {}
        for p_c in config["sweep_m"]["p_c_list"]:
            frac_data[f'frac_nodes_pc{p_c}'] = results_m['frac_nodes_kept'][p_c]
            frac_data[f'frac_edges_pc{p_c}'] = results_m['frac_edges_kept'][p_c]
            frac_data[f'valid_pc{p_c}']       = [not x for x in results_m['no_window'][p_c]]
        fm.export_grid_to_csv(frac_data, f'sweep_m_edges_frac_{gof_theory}', metric_name='frac',
            x_values=results_m['m_values'], x_name='m_edges',
            metadata_dict={'Sweep': 'm_edges', 'GoF theory': gof_theory, 'MLE type': mle_theory,
                           'Node type': config["sweep_m"]["node_type"], 'a': config["sweep_m"]["a"]})
        v1d = GoFDiagnostics.CSN1DVis(gof)
        fig_nodes = v1d.plot_frac_sweep(results_m['m_values'], results_m['frac_nodes_kept'],
            results_m['frac_edges_kept'], results_m['no_window'], config["sweep_m"]["p_c_list"],
            'm_edges', config["sweep_m"]["node_type"], config["sweep_m"]["a"], metric='nodes')
        fm.save_fig_with_suffix(fig_nodes, "sweep_m_edges_frac_nodes", gof_theory)
        fig_edges = v1d.plot_frac_sweep(results_m['m_values'], results_m['frac_nodes_kept'],
            results_m['frac_edges_kept'], results_m['no_window'], config["sweep_m"]["p_c_list"],
            'm_edges', config["sweep_m"]["node_type"], config["sweep_m"]["a"], metric='edges')
        fm.save_fig_with_suffix(fig_edges, "sweep_m_edges_frac_edges", gof_theory)
        num_p_vs_b_m = plots.get("csn_p_vs_b_m", 0)
        if num_p_vs_b_m > 0:
            diag_indices = gof._select_indices_for_diagnostics(len(results_m['m_values']), num_p_vs_b_m)
            for idx in diag_indices:
                x_val    = int(results_m['m_values'][idx])
                scan_res = results_m['windows_info'][idx]
                if not scan_res.get('windows', []):
                    continue
                fig_p = v1d.plot_p_vs_b(scan_res, config["sweep_m"]["p_c_list"], x_val,
                    'm_edges', config["sweep_m"]["node_type"], config["sweep_m"]["a"], z=1.0)
                fm.save_fig_with_suffix(fig_p, f"p_vs_b_m_edges_{x_val}", gof_theory)

    if plots["sweep_n0_csn"]:
        results_n0 = gof.csn_sweep_n0(net, **config["sweep_n0"],
                                        mle_type=mle_theory)
        fm.save_fig_with_suffix(results_n0["fig"], "sweep_n0_csn", gof_theory)
        frac_data = {}
        for p_c in config["sweep_n0"]["p_c_list"]:
            frac_data[f'frac_nodes_pc{p_c}'] = results_n0['frac_nodes_kept'][p_c]
            frac_data[f'frac_edges_pc{p_c}'] = results_n0['frac_edges_kept'][p_c]
            frac_data[f'valid_pc{p_c}']       = [not x for x in results_n0['no_window'][p_c]]
        fm.export_grid_to_csv(frac_data, f'sweep_n0_frac_{gof_theory}', metric_name='frac',
            x_values=results_n0['n0_values'], x_name='n0',
            metadata_dict={'Sweep': 'n0', 'GoF theory': gof_theory, 'MLE type': mle_theory,
                           'Node type': config["sweep_n0"]["node_type"], 'a': config["sweep_n0"]["a"]})
        v1d = GoFDiagnostics.CSN1DVis(gof)
        fig_nodes = v1d.plot_frac_sweep(results_n0['n0_values'], results_n0['frac_nodes_kept'],
            results_n0['frac_edges_kept'], results_n0['no_window'], config["sweep_n0"]["p_c_list"],
            'n0', config["sweep_n0"]["node_type"], config["sweep_n0"]["a"], metric='nodes')
        fm.save_fig_with_suffix(fig_nodes, "sweep_n0_frac_nodes", gof_theory)
        fig_edges = v1d.plot_frac_sweep(results_n0['n0_values'], results_n0['frac_nodes_kept'],
            results_n0['frac_edges_kept'], results_n0['no_window'], config["sweep_n0"]["p_c_list"],
            'n0', config["sweep_n0"]["node_type"], config["sweep_n0"]["a"], metric='edges')
        fm.save_fig_with_suffix(fig_edges, "sweep_n0_frac_edges", gof_theory)

    if plots["grid_2d_sweep"]:
        _allowed_keys = {"m_min","m_max","m_step","n0_min","n0_max","n0_step","node_type","a",
                         "p_c_list","N_sims","b_grid_type","n_b","b_min","b_max","z"}
        _grid2d_args = {k: v for k, v in config["grid_2d"].items() if k in _allowed_keys}
        grid_results = gof.csn_sweep_2d_grid(net, **_grid2d_args, mle_type=mle_theory)
        m_values  = grid_results["m_values"]
        n0_values = grid_results["n0_values"]
        p_c_list  = grid_results["p_c_list"]
        v2d = GoFDiagnostics.CSN2DVis(gof)
        for p_c in p_c_list:
            margin_data = grid_results["p_value_grid"][p_c] - p_c
            fig = v2d.plot_contour_generic(m_values, n0_values, margin_data,
                metric_label=r'$M = p^* - p_c$', use_csn_cmap=False)
            fm.save_fig_with_suffix(fig, f"margin_pc{p_c}", gof_theory)
            fm.export_grid_to_csv(margin_data, f"margin_pc{p_c}_{gof_theory}",
                metric_name='margin', m_values=m_values, n0_values=n0_values, p_c=p_c)
            fig = v2d.plot_contour_generic(m_values, n0_values, grid_results["frac_nodes_grid"][p_c],
                metric_label=r'$F_{\mathrm{nodes}}^*$', use_csn_cmap=True)
            fm.save_fig_with_suffix(fig, f"frac_nodes_pc{p_c}", gof_theory)
            fm.export_grid_to_csv(grid_results["frac_nodes_grid"][p_c],
                f"frac_nodes_pc{p_c}_{gof_theory}", metric_name='frac_nodes',
                m_values=m_values, n0_values=n0_values, p_c=p_c)
            fig = v2d.plot_contour_generic(m_values, n0_values, grid_results["frac_edges_grid"][p_c],
                metric_label=r'$F_{\mathrm{edges}}^*$', use_csn_cmap=True)
            fm.save_fig_with_suffix(fig, f"frac_edges_pc{p_c}", gof_theory)
            fm.export_grid_to_csv(grid_results["frac_edges_grid"][p_c],
                f"frac_edges_pc{p_c}_{gof_theory}", metric_name='frac_edges',
                m_values=m_values, n0_values=n0_values, p_c=p_c)

    fm.finalize_metadata()
    print(f"\nAll outputs saved to: {fm.path()}")