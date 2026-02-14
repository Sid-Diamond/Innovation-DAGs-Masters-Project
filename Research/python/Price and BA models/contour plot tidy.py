import numpy as np
import networkx as nx
import matplotlib
from mpl_toolkits.mplot3d import Axes3D 
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
from scipy.special import gamma as gamma_func, betaln
from typing import Dict, Tuple, List
import time
from scipy.optimize import minimize
from pathlib import Path
import datetime
import json

class FileManager:

    def __init__(self, config: dict, base: str = "runs"):
        self.config = config
        self.base = Path(base)
        self.base.mkdir(exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base / f"run_{ts}"
        self.run_dir.mkdir()
        
        # Track start time for finalization
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

    def path(self):
        return self.run_dir
    
class DirectedHomophilicNetwork:
    """Optimized directed network with homophilic preferential attachment."""

    def __init__(self,n0: int,n_nodes: int,m_edges: int,h: float,f_a: float,mu_a: float,mu_b: float,
        seed: int = None,build_graph: bool = True, ):
        
        # Network parameters
        self.n0, self.n_nodes, self.m_edges = n0, n_nodes, m_edges
        self.h, self.f_a, self.f_b = h, f_a, 1 - f_a
        self.mu = {'a': mu_a, 'b': mu_b}
        self.seed = seed
        self.build_graph = build_graph   # <--- STORE FLAG

        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Computed parameters
        self.lambda_a = h * f_a + (1 - f_a) * (1 - h)
        self.lambda_b = h * self.f_b + (1 - self.f_b) * (1 - h)
        self.lambda_ = {'a': self.lambda_a, 'b': self.lambda_b}

        # Network state
        self.graph = None
        self.node_types = None
        self.edge_evolution = []
        self.g_a, self.g_b, self.g_b_empirical, self.Z_factor, self.Z_tilde = (None,None,None,None,None,)

        # Cached state for speed
        self.in_degrees = None
        self.in_edges_a_count = 0
        self.in_edges_b_count = 0
        self._cdf_cache: dict = {}

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
        for source in range(self.n0):
            # select m targets without self-loops or duplicates
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

    def _fit_asymptotes(self, fraction: float = 0.05):
        """Fit asymptotic g values from evolution data."""
        mean_deg_a = np.array([d['in_edges_a'] / d['t'] for d in self.edge_evolution])
        mean_deg_b = np.array([d['in_edges_b'] / d['t'] for d in self.edge_evolution])

        n_tail = max(1, int(len(mean_deg_a) * fraction))
        g_a = mean_deg_a[-n_tail:].mean()
        g_b_empirical = mean_deg_b[-n_tail:].mean()  # Store for comparison
        g_b = self.m_edges - g_a  # Enforce constraint

        self._compute_theoretical_params(g_a, g_b, g_b_empirical)

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

    def theoretical_distribution(self, k, node_type: str):
        """
        Theoretical in-degree distribution with analytic continuation.
        """
        beta = self.get_beta_for_type(node_type)
        pmf = self.pmf_from_beta(k, beta)
        return pmf.item() if np.ndim(pmf) == 1 and pmf.size == 1 else pmf

    def cdf_from_beta_truncated(self, k, beta, kmin: int, kmax: int) -> np.ndarray:
        """
        Conditional CDF F(k | beta, kmin <= K <= kmax) on integer support
        kmin,...,kmax, constructed from pmf_from_beta and renormalised
        over [kmin, kmax].
        """
        k = np.atleast_1d(k)
        k_int = np.floor(k).astype(int)

        # Normalise beta to a hashable key (round to avoid small float noise)
        beta = np.asarray(beta, dtype=float)
        beta_key = tuple(np.round(beta, decimals=12))  # (p0, alpha, gamma)
        cache_key = (beta_key, int(kmin), int(kmax))

        # Build and cache support-CDF table if needed
        if cache_key not in self._cdf_cache:
            support = np.arange(kmin, kmax + 1, dtype=int)
            pmf = self.pmf_from_beta(support, beta).astype(float)

            Z = pmf.sum()
            if Z <= 0:
                # Degenerate case: CDF is 0 until kmax, then 1
                cdf_support = np.ones_like(support, dtype=float)
            else:
                pmf /= Z
                cdf_support = np.cumsum(pmf)

            self._cdf_cache[cache_key] = (support, cdf_support)
        else:
            support, cdf_support = self._cdf_cache[cache_key]

        # Answer the CDF query for each k_int via the cached table
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

class GoFDiagnostics:

    def theoretical_cdf_discrete(self, net: DirectedHomophilicNetwork, k, node_type: str, kmin: int, kmax: int, beta=None,):
        k = np.atleast_1d(k)

        if beta is None:
            beta = net.get_beta_for_type(node_type)

        return net.cdf_from_beta_truncated(k, beta, kmin, kmax)

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
   
    def _fit_beta_mle(self, net: DirectedHomophilicNetwork, data, a: int, b: int, node_type: str, beta_init):
        """
        MLE for beta = (p0, alpha, gamma) on truncated window [a, b].
        Uses formal MLE with precomputed support and efficient likelihood computation.
        """
        
        data = np.asarray(data, dtype=int)
        if data.size == 0:
            params0 = net._get_params(node_type)
            beta_mle = np.array([params0['p0'], params0['alpha'], params0['gamma']], dtype=float)
            return beta_mle, True

        params0 = net._get_params(node_type)

        if beta_init is None:
            p0_0 = float(params0['p0'])
            alpha0 = float(params0['alpha'])
            gamma0 = float(params0['gamma'])
        else:
            p0_0, alpha0, gamma0 = np.asarray(beta_init, dtype=float)

        k_support = np.arange(a, b + 1, dtype=int)
        unique_k, counts = np.unique(data, return_counts=True)

        mask_window = (unique_k >= a) & (unique_k <= b)
        unique_k = unique_k[mask_window]
        counts = counts[mask_window]

        if counts.size == 0:
            return np.array([p0_0, alpha0, gamma0], dtype=float), True

        k_to_idx = {k: i for i, k in enumerate(k_support)}
        idxs = np.array([k_to_idx[k] for k in unique_k], dtype=int)

        zero_mask = (k_support == 0)
        pos_mask = (k_support > 0)
        k_pos = k_support[pos_mask]

        def objective(theta: np.ndarray) -> float:
            """Negative log-likelihood for truncated distribution."""
            p0, alpha, gamma = theta

            pmf = np.zeros_like(k_support, dtype=float)
            
            if np.any(zero_mask):
                pmf[zero_mask] = p0
            
            if np.any(pos_mask):
                log_ratio = betaln(k_pos, alpha + gamma) - betaln(k_pos, alpha)
                pmf[pos_mask] = p0 * np.exp(log_ratio)

            Z_trunc = pmf.sum()
            if Z_trunc <= 0:
                return 1e10

            pmf_norm = pmf / Z_trunc
            pmf_norm = np.clip(pmf_norm, 1e-300, 1.0)

            pmf_at_data = pmf_norm[idxs]
            log_pmf_at_data = np.log(pmf_at_data)
            
            nll = -np.sum(counts * log_pmf_at_data)
            return nll

        theta0 = np.array([p0_0, alpha0, gamma0], dtype=float)
        bounds = [(1e-6, 1.0 - 1e-6), (1e-4, None), (1e-4, None)]

        res = minimize(objective, theta0, method='L-BFGS-B', bounds=bounds)

        if not res.success:
            p0_mle, alpha_mle, gamma_mle = p0_0, alpha0, gamma0
            used_fallback = True
        else:
            p0_mle, alpha_mle, gamma_mle = res.x
            used_fallback = False

        beta_mle = np.array([p0_mle, alpha_mle, gamma_mle], dtype=float)
        return beta_mle, used_fallback

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

    def mc_on_window(self,net: DirectedHomophilicNetwork,node_type: str,a: int,b: int,beta_theory,N_sims: int,data_cache=None,):
        """
        Monte Carlo on window [a, b].
        """
        D_theory_list = []
        D_mle_list = []
        beta_mle_list = []
        fallback_flags = []

        beta_theory_arr = np.asarray(beta_theory) if beta_theory is not None else None

        # Prepare data cache with proper validation
        if data_cache is None:
            # Generate fresh simulations only if cache not provided
            print(f"  Generating {N_sims} network simulations for MC...")
            cache_list = []
            for sim_idx in range(N_sims):
                net_sim = DirectedHomophilicNetwork(n0=net.n0,n_nodes=net.n_nodes,m_edges=net.m_edges,h=net.h,f_a=net.f_a,
                    mu_a=net.mu['a'],  mu_b=net.mu['b'],seed=None, )
                net_sim.generate_network()
                degrees = np.asarray(net_sim._get_degrees(node_type), dtype=int)
                cache_list.append(degrees)
        else:
            # Use provided cache with validation
            if isinstance(data_cache, np.ndarray):
                # Single array: wrap in list
                cache_list = [np.asarray(data_cache, dtype=int)]
            elif isinstance(data_cache, list):
                # List of arrays: validate each
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

        # Evaluate MC statistics for each sample
        for sim_idx, degrees in enumerate(cache_list):
            # Filter to window [a, b]
            data_s = degrees[(degrees >= a) & (degrees <= b)]

            if data_s.size == 0:
                # No data in window
                D_theory_list.append(0.0)
                D_mle_list.append(0.0)
                beta_mle_list.append(
                    beta_theory_arr if beta_theory_arr is not None 
                    else np.array([np.nan, np.nan, np.nan])
                )
                fallback_flags.append(True)
                continue

            # Compute D_theory using provided beta_theory
            D_theory_s = self.csn_distance(net, data_s, a, b, beta_theory, node_type)

            # Fit MLE and compute D_mle
            beta_mle_s, used_fallback_s = self._fit_beta_mle(
                net, data_s, a, b, node_type, beta_init=None
            )
            D_mle_s = self.csn_distance(net, data_s, a, b, beta_mle_s, node_type)

            D_theory_list.append(D_theory_s)
            D_mle_list.append(D_mle_s)
            beta_mle_list.append(beta_mle_s)
            fallback_flags.append(used_fallback_s)

        # Aggregate statistics
        D_theory_arr = np.array(D_theory_list, dtype=float)
        D_mle_arr = np.array(D_mle_list, dtype=float)
        fallback_flags = np.asarray(fallback_flags, dtype=bool)

        # Probability that D_mle > D_theory
        greater_mask = D_mle_arr > D_theory_arr
        p = float(np.mean(greater_mask.astype(float)))
        sigma_p = float(np.sqrt(p * (1.0 - p) / max(len(D_theory_arr), 1)))

        try:
            beta_mle_arr = np.asarray(beta_mle_list, dtype=float)
        except (ValueError, TypeError):
            beta_mle_arr = np.array(beta_mle_list, dtype=object)

        frac_moved = float(np.mean(~fallback_flags)) if fallback_flags.size > 0 else np.nan

        beta_mle_mean = None
        beta_mle_std = None
        if (
            isinstance(beta_mle_arr, np.ndarray)
            and beta_mle_arr.ndim == 2
            and beta_mle_arr.shape[0] == fallback_flags.size
        ):
            mask_non_fallback = ~fallback_flags
            if np.any(mask_non_fallback):
                beta_mle_non_fallback = beta_mle_arr[mask_non_fallback]
                beta_mle_mean = np.nanmean(beta_mle_non_fallback, axis=0)
                beta_mle_std = np.nanstd(beta_mle_non_fallback, axis=0)

        percent_diff_mean_vs_theory = None
        if beta_theory_arr is not None and beta_mle_mean is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                percent_diff_mean_vs_theory = 100.0 * (
                    (beta_mle_mean - beta_theory_arr) / beta_theory_arr
                )
                percent_diff_mean_vs_theory = np.where(
                    beta_theory_arr == 0.0, np.nan, percent_diff_mean_vs_theory
                )

        result = {
            'a': a,
            'b': b,
            'D_theory': D_theory_arr,
            'D_mle': D_mle_arr,
            'beta_mle': beta_mle_arr,
            'p': p,
            'sigma_p': sigma_p,
            'beta_mle_mean': beta_mle_mean,
            'beta_mle_std': beta_mle_std,
            'beta_theory': beta_theory_arr,
            'frac_moved': frac_moved,
            'percent_diff_mean_vs_theory': percent_diff_mean_vs_theory,
        }
        return result

    def scan_over_b(self,net: DirectedHomophilicNetwork,node_type: str,a: int,beta_theory,candidate_bs=None,
        N_sims: int = 20,b_min: int = None,b_max: int = None,n_b: int = 20,b_grid_type: str = 'linear',data_cache: np.ndarray = None):
        """
        Scan over b values. If data_cache is None, generate one set of N_sims
        non-truncated samples (degree arrays) and reuse across all b values.
        """
        bs = self._build_b_grid(net=net,node_type=node_type,a=a,candidate_bs=candidate_bs,
            b_min=b_min,b_max=b_max,n_b=n_b,b_grid_type=b_grid_type)

        windows_results = []

        if bs is None or len(bs) == 0:
            return {
                'a': a,
                'beta_theory': beta_theory,
                'windows': [],
                'b_grid': None,
            }


        if data_cache is None:
            data_cache_list = []
            for _ in range(N_sims):
                net_sim = DirectedHomophilicNetwork(n0=net.n0,n_nodes=net.n_nodes, m_edges=net.m_edges, h=net.h,
                    f_a=net.f_a, mu_a=net.mu['a'],mu_b=net.mu['b'],seed=None)
                net_sim.generate_network()
                degrees = np.array(net_sim._get_degrees(node_type), dtype=int)
                data_cache_list.append(degrees)
            data_cache = data_cache_list
        else:
            if not isinstance(data_cache, list):
                data_cache = [np.asarray(data_cache, dtype=int)]

        # Now evaluate each candidate b by delegating to mc_on_window
        for b in bs:
            if b <= a:
                raise ValueError(f"Each b_j must satisfy b_j > a. Got a={a}, b_j={b}.")

            res_b = self.mc_on_window(net=net,node_type=node_type,a=a,
                b=int(b),beta_theory=beta_theory,N_sims=len(data_cache), data_cache=data_cache)

            window_result = {
                'a': a,
                'b': int(b),
                'D_theory': res_b['D_theory'],
                'D_mle': res_b['D_mle'],
                'beta_mle': res_b['beta_mle'],
                'p': res_b['p'],
                'sigma_p': res_b['sigma_p'],
                'beta_mle_mean': res_b['beta_mle_mean'],
                'beta_mle_std': res_b['beta_mle_std'],
                'beta_theory': res_b['beta_theory'],
                'frac_moved': res_b['frac_moved'],
                'percent_diff_mean_vs_theory': res_b['percent_diff_mean_vs_theory'],
            }
            windows_results.append(window_result)

        result = {
            'a': a,
            'beta_theory': beta_theory,
            'windows': windows_results,
            'b_grid': np.array(bs, dtype=int),
        }
        return result

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

    def _csn_sweep_core(self, base_net: DirectedHomophilicNetwork, sweep_param_values: np.ndarray, vary: str, node_type: str, a: int,
        candidate_bs,b_min: int, b_max: int, n_b: int, b_grid_type: str, N_sims: int, p_c_list,figsize=(12, 6),):
        
        plt.rcParams['font.family'] = 'Times New Roman'

        windows_info = []
        frac_nodes_kept = {p_c: [] for p_c in p_c_list}
        frac_edges_kept = {p_c: [] for p_c in p_c_list}
        no_window = {p_c: [] for p_c in p_c_list}

        label = 'm_edges' if vary == 'm_edges' else 'n0'
        print(f"\nCSN sweep over {label}: {sweep_param_values}")

        for val in sweep_param_values:
            if vary == 'm_edges':
                n0 = base_net.n0
                n_nodes = base_net.n_nodes
                m_edges = int(val)
            elif vary == 'n0':
                n0 = int(val)
                n_nodes = base_net.n_nodes
                m_edges = base_net.m_edges
            else:
                raise ValueError("vary must be 'm_edges' or 'n0'")

            print(f"\n  {label} = {val}: generating network and scanning windows...")

            net_temp = DirectedHomophilicNetwork(n0=n0, n_nodes=n_nodes, m_edges=m_edges, h=base_net.h, f_a=base_net.f_a, mu_a=base_net.mu['a'],
                mu_b=base_net.mu['b'],)
            
            net_temp.generate_network()
            degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
            beta_theory = net_temp.get_beta_for_type(node_type)

            scan_res = self.scan_over_b(net=net_temp, node_type=node_type, a=a, beta_theory=beta_theory, candidate_bs=candidate_bs,
                N_sims=N_sims, b_min=b_min, b_max=b_max,n_b=n_b, b_grid_type=b_grid_type,)

            # store degrees so diagnostics can reconstruct N(b)
            scan_res['degrees_for_node_type'] = degrees.copy()
            windows_info.append(scan_res)

            windows = scan_res['windows']
            bs = scan_res['b_grid']

            if bs is None or len(bs) == 0 or not windows:
                print("    No valid b range; skipping.")
                for p_c in p_c_list:
                    frac_nodes_kept[p_c].append(0.0)
                    frac_edges_kept[p_c].append(0.0)
                    no_window[p_c].append(True)
                continue

            if len(bs) > 6:
                grid_str = f"[{bs[0]}, {bs[1]},... {bs[-2]}, {bs[-1]}]"
            else:
                grid_str = str(bs.tolist())
            print(f"    b-grid (candidate b_j): {grid_str}")


            full_w = max(windows, key=lambda w: w['b'])
            print(f"    no-trunc p={full_w['p']:.4f} ± {full_w['sigma_p']:.4f}")

            total_edges = degrees.sum() if degrees.size > 0 else 0

            pcs_results = self.select_largest_window_for_pcs(
                windows=windows, a=a, p_c_list=p_c_list,)

            for p_c in p_c_list:
                lw = pcs_results[p_c]['largest_window']

                if lw is None:
                    frac_nodes_kept[p_c].append(0.0)
                    frac_edges_kept[p_c].append(0.0)
                    no_window[p_c].append(True)
                    print(f"    p_c={p_c}: no acceptable window")
                    continue

                b_star = lw['b']
                p_val = lw['p']
                sigma_p = lw['sigma_p']

                mask_keep = (degrees >= a) & (degrees <= b_star)
                fn = mask_keep.mean()
                fe = degrees[mask_keep].sum() / total_edges if total_edges > 0 else 0.0

                frac_nodes_kept[p_c].append(fn)
                frac_edges_kept[p_c].append(fe)
                no_window[p_c].append(False)

                print(
                    f"    p_c={p_c}: b*={b_star}, "
                    f"p-z*σ={p_val - sigma_p:.4f}, "
                    f"nodes={fn:.4f}, edges={fe:.4f}"
                )


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        if vary == 'm_edges':
            x_values = sweep_param_values
            x_label = 'm_edges'
            frac_new_nodes = base_net.n_nodes / (base_net.n_nodes + base_net.n0)
        else:
            x_values = sweep_param_values
            x_label = 'n0'
            frac_new_nodes = base_net.n_nodes / (base_net.n_nodes + sweep_param_values)

        colors = plt.cm.viridis(np.linspace(0, 1, len(p_c_list)))

        for color, p_c in zip(colors, p_c_list):
            fn_arr = np.array(frac_nodes_kept[p_c], dtype=float)
            nw_arr = np.array(no_window[p_c], dtype=bool)

            ok_mask = ~nw_arr
            fail_mask = nw_arr

            ax1.plot(
                x_values[ok_mask],
                fn_arr[ok_mask],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=5,
                color=color,
                label=f'p_c = {p_c}',
            )

            ax1.scatter(
                x_values[fail_mask],
                np.zeros_like(fn_arr[fail_mask]),
                marker='x',
                s=60,
                color=color,
                zorder=3,
            )

        if np.isscalar(frac_new_nodes):
            ax1.axhline(
                frac_new_nodes,
                linestyle='--',
                color='black',
                linewidth=1.5,
            )

        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Fraction of nodes kept')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        for color, p_c in zip(colors, p_c_list):
            fe_arr = np.array(frac_edges_kept[p_c], dtype=float)
            nw_arr = np.array(no_window[p_c], dtype=bool)

            ok_mask = ~nw_arr
            fail_mask = nw_arr

            ax2.plot(
                x_values[ok_mask],
                fe_arr[ok_mask],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=5,
                color=color,
                label=f'p_c = {p_c}',
            )

            ax2.scatter(
                x_values[fail_mask],
                np.zeros_like(fe_arr[fail_mask]),
                marker='x',
                s=50,
                color=color,
                zorder=3,
            )

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
        }

    def csn_sweep_m_edges(self, net: DirectedHomophilicNetwork, m_min: int, m_max: int, m_step: int = 1, node_type: str = 'b', a: int = 0,
        candidate_bs=None, b_min: int = None, b_max: int = None, n_b: int = 20, b_grid_type: str = 'linear', N_sims: int = 20,
        p_c_list=(0.1, 0.2, 0.4, 0.6), figsize=(12, 6),):
        m_values = np.arange(m_min, m_max + 1, m_step, dtype=int)

        res = self._csn_sweep_core( base_net=net, sweep_param_values=m_values, vary='m_edges',node_type=node_type, a=a, candidate_bs=candidate_bs,
            b_min=b_min, b_max=b_max, n_b=n_b, b_grid_type=b_grid_type, N_sims=N_sims, p_c_list=p_c_list, figsize=figsize,)

        return {
            'm_values': res['x_values'],
            'frac_nodes_kept': res['frac_nodes_kept'],
            'frac_edges_kept': res['frac_edges_kept'],
            'no_window': res['no_window'],
            'windows_info': res['windows_info'],
            'fig': res['fig'],
        }

    def csn_sweep_n0(self, net: DirectedHomophilicNetwork, n0_min: int, n0_max: int, n0_step: int = 1, node_type: str = 'b', a: int = 0,
        candidate_bs=None, b_min: int = None, b_max: int = None, n_b: int = 20, b_grid_type: str = 'linear', N_sims: int = 20,
        p_c_list=(0.1, 0.2, 0.4, 0.6), figsize=(12, 6),):
        n0_values = np.arange(n0_min, n0_max + 1, n0_step, dtype=int)

        res = self._csn_sweep_core(base_net=net, sweep_param_values=n0_values, vary='n0', node_type=node_type, a=a,
            candidate_bs=candidate_bs, b_min=b_min, b_max=b_max, n_b=n_b, b_grid_type=b_grid_type,N_sims=N_sims, p_c_list=p_c_list,
            figsize=figsize,)

        return {
            'n0_values': res['x_values'],
            'frac_nodes_kept': res['frac_nodes_kept'],
            'frac_edges_kept': res['frac_edges_kept'],
            'no_window': res['no_window'],
            'windows_info': res['windows_info'],
            'fig': res['fig'],
        }
    
    def _select_indices_for_diagnostics(self, n: int, k: int):

        if k <= 0 or n <= 0:
            return []
        if k >= n:
            return list(range(n))

        positions = np.linspace(0, n - 1, k)
        indices = sorted(set(int(round(p)) for p in positions))
        return indices

    def plot_p_vs_b_diagnostic_combined(self, scan_res, p_c_list, z: float = 1.0, title_prefix: str = "", figsize=(8, 5), x_context: str = None, ):
        windows = scan_res.get('windows', [])
        if not windows:
            print("No windows to plot in p_vs_b diagnostic.")
            return None

        bs = np.array([w['b'] for w in windows], dtype=int)
        ps = np.array([w['p'] for w in windows], dtype=float)
        sigmas = np.array([w['sigma_p'] for w in windows], dtype=float)
        p_lower = ps - z * sigmas

        fig, ax = plt.subplots(figsize=figsize)

        # Base vertical error 'bar' as a gray line (no caps)
        ax.errorbar(
            bs,
            ps,
            yerr=sigmas,
            fmt='none',
            ecolor='0.6',
            elinewidth=1.5,
            capsize=0,
            zorder=1,
        )

        tick_color = 'orange'
        tick_size = 0.01  # 

        for b, p_val, s in zip(bs, ps, sigmas):
            top = p_val + s
            mid = p_val
            bot = p_val - s

            y_min, y_max = -0.05, 1.05
            tick_half = tick_size * (y_max - y_min) / 2.0

            for y in (top, mid, bot):
                ax.plot(
                    [b - 0.1, b + 0.1],
                    [y, y],
                    color=tick_color,
                    linewidth=1.5,
                    zorder=2,
                )

        ax.plot(
            bs,
            p_lower,
            '-s',
            color='C1',
            linewidth=2,
            markersize=5,
            label=fr'Lower bound $p(b) - {z}\,\sigma_p$',
            zorder=3,
        )

        colors_pc = plt.cm.plasma(np.linspace(0, 1, len(p_c_list)))
        for color_pc, p_c in zip(colors_pc, p_c_list):
            ax.axhline(
                y=p_c,
                color=color_pc,
                linestyle='--',
                linewidth=1.5,
                label=fr'$p_c = {p_c}$',
            )

        a = scan_res.get('a', None)
        if x_context is not None:
            main_title = f"modified CSN  P value vs data truncation for {x_context}"
        else:
            main_title = "modified CSN  P value vs data truncation"

        parts = [main_title]
        if a is not None:
            parts.append(f"[a = {a}]")
        if title_prefix:
            parts.append(title_prefix)

        ax.set_title(" | ".join(parts))

        ax.set_xlabel(r'$b$ (upper truncation)')
        ax.set_ylabel(r'$p(b)$ and lower bounds')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        plt.tight_layout()
        return fig

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

    def csn_diagnostic_plots_over_sweep(self, sweep_results: dict, p_c_list, num_p_vs_b: int, fm: "FileManager", node_type: str, sweep_label: str,  z: float = 1.0, a: int = 0,):
    
        if num_p_vs_b <= 0:
            return

        if sweep_label == 'm_edges':
            x_values = np.asarray(sweep_results['m_values'], dtype=int)
        else:
            x_values = np.asarray(sweep_results['n0_values'], dtype=int)

        windows_info = sweep_results['windows_info']
        n = len(x_values)
        if n == 0:
            return

        diag_indices = self._select_indices_for_diagnostics(n=n, k=num_p_vs_b)

        for idx in diag_indices:
            x_val = int(x_values[idx])
            scan_res = windows_info[idx]
            windows = scan_res.get('windows', [])
            if not windows:
                continue

            pcs_results = self.select_largest_window_for_pcs(
                windows=windows,
                a=a,
                p_c_list=p_c_list,
            )
            if sweep_label == 'm_edges':
                x_context = rf"$m = {x_val}$"
            else:
                x_context = rf"$n_0 = {x_val}$"

            # Build figure without overlay first
            fig_diag = self.plot_p_vs_b_diagnostic_combined(scan_res=scan_res, p_c_list=p_c_list,
            z=z, title_prefix=f"node_type={node_type}", figsize=(8, 5), x_context=x_context,)

            # overlay p(d_e, N) curve from D_theory
            degrees_arr = scan_res.get('degrees_for_node_type', None)
            if degrees_arr is not None and fig_diag is not None:
                # minimal dummy "net" with only _get_degrees implemented
                class _DummyNet:
                    def __init__(self, degrees):
                        self._degrees = np.asarray(degrees, dtype=int)
                    def _get_degrees(self, nt):
                        return self._degrees

                dummy_net = _DummyNet(degrees_arr)
                bs_extra, p_extra = self._p_from_D_theory_over_b(
                    scan_res=scan_res,
                    net=dummy_net,
                    node_type=node_type,
                )
                if bs_extra.size > 0:
                    ax = fig_diag.axes[0]
                    ax.plot(
                        bs_extra,
                        p_extra,
                        '-^',
                        color='C3',
                        linewidth=2,
                        markersize=4,
                        label=r'$p(d_e, N)$ from $D_{\mathrm{theory}}$',
                    )
                    ax.legend(loc='best')
     
            if fig_diag is not None:
                fm.save_fig(
                    fig_diag,
                    f"p_vs_b_{sweep_label}{x_val}",
                ) 

    def csn_sweep_2d_grid(self, net: DirectedHomophilicNetwork, m_min: int, m_max: int, m_step: int, n0_min: int, n0_max: int, n0_step: int, 
                          node_type: str = 'b', a: int = 0, candidate_bs=None, b_min: int = None, b_max: int = None, n_b: int = 20, 
                          b_grid_type: str = 'linear', N_sims: int = 20, p_c_list=(0.1, 0.2, 0.4, 0.6), z: float = 1.0,):
        """2D sweep over m_edges and n0. Skips invalid region where n0 <= m."""
        m_values = np.arange(m_min, m_max + 1, m_step, dtype=int)
        n0_values = np.arange(n0_min, n0_max + 1, n0_step, dtype=int)
        
        frac_nodes_grid = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
        frac_edges_grid = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
        b_star_grid = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
        p_value_grid = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
        
        print(f"\n2D Grid: m ∈ [{m_min}, {m_max}], n0 ∈ [{n0_min}, {n0_max}]")
        
        total_points = len(n0_values) * len(m_values)
        valid_points = 0
        
        for i, n0 in enumerate(n0_values):
            for j, m in enumerate(m_values):
                if n0 <= m:
                    continue
                
                valid_points += 1
                print(f"\n  Point {valid_points}/{total_points}: (m={m}, n0={n0})")
                
                net_temp = DirectedHomophilicNetwork(n0=n0, n_nodes=net.n_nodes, m_edges=m, h=net.h, f_a=net.f_a, mu_a=net.mu['a'], mu_b=net.mu['b'])
                net_temp.generate_network()
                
                degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                if degrees.size == 0:
                    continue
                
                beta_theory = net_temp.get_beta_for_type(node_type)
                scan_res = self.scan_over_b(net=net_temp,node_type=node_type,a=a,beta_theory=beta_theory,
                    candidate_bs=candidate_bs, N_sims=N_sims,b_min=b_min,b_max=b_max,n_b=n_b,b_grid_type=b_grid_type)

                windows = scan_res['windows']
                if not windows:
                    continue

                full_w = max(windows, key=lambda w: w['b'])
                print(f"    no-trunc p={full_w['p']:.4f} ± {full_w['sigma_p']:.4f}")

                pcs_results = self.select_largest_window_for_pcs(windows=windows, a=a, p_c_list=p_c_list)
                total_edges = degrees.sum() if degrees.size > 0 else 0
                
                for p_c in p_c_list:
                    lw = pcs_results[p_c]['largest_window']
                    if lw is None:
                        frac_nodes_grid[p_c][i, j] = 0.0
                        frac_edges_grid[p_c][i, j] = 0.0
                        continue
                    
                    b_star = lw['b']
                    p_val = lw['p']
                    sigma_p = lw['sigma_p']
                    lower_bound = p_val - z * sigma_p
                    
                    mask_keep = (degrees >= a) & (degrees <= b_star)
                    fn = mask_keep.mean()
                    fe = degrees[mask_keep].sum() / total_edges if total_edges > 0 else 0.0
                    
                    frac_nodes_grid[p_c][i, j] = fn
                    frac_edges_grid[p_c][i, j] = fe
                    b_star_grid[p_c][i, j] = b_star
                    p_value_grid[p_c][i, j] = lower_bound
                    
                    print(f"    p_c={p_c}: b*={b_star}, p-z*σ={lower_bound:.4f}, nodes={fn:.4f}, edges={fe:.4f}")
        
        return {'m_values': m_values, 'n0_values': n0_values, 'frac_nodes_grid': frac_nodes_grid, 'frac_edges_grid': frac_edges_grid,
                'b_star_grid': b_star_grid, 'p_value_grid': p_value_grid, 'p_c_list': p_c_list, 'z': z, 'a': a, 'node_type': node_type}

    class CSN2DVis:
        """
        2D visualization utilities for CSN-based windows.
        Implements:
        - Hard-acceptance plots (nodes/edges) from csn_sweep_2d_grid
        - A) Soft acceptance coverage surfaces (C_soft, C_max) for each p_c in p_c_list
        - B) Margin map M = p* - p_c with F* overlay for each p_c in p_c_list
        """

        def __init__(self, parent_gof: "GoFDiagnostics"):
            self.gof = parent_gof  # access scan_over_b and other helpers

        # ---------- Core series builders ----------

        def build_series_for_scan(self, scan_res: dict, degrees: np.ndarray, a: int, z: float = 1.0):
            """
            From a single scan_over_b result, build arrays over b:
            - bs: candidate upper truncations
            - F(b): fraction of nodes kept
            - p(b): estimated p
            - sigma(b): std of p
            - L(b) = p(b) - z*sigma(b)  (conservative lower bound)
            """
            windows = scan_res.get('windows', [])
            if not windows:
                return (np.array([]),) * 5

            bs = np.array([w['b'] for w in windows], dtype=int)
            p = np.array([w['p'] for w in windows], dtype=float)
            sigma = np.array([w['sigma_p'] for w in windows], dtype=float)
            L = p - z * sigma

            degrees = np.asarray(degrees, dtype=int)
            N_tot = max(degrees.size, 1)
            F = np.zeros_like(bs, dtype=float)
            for i, b in enumerate(bs):
                mask = (degrees >= a) & (degrees <= b)
                F[i] = mask.mean() if N_tot > 0 else 0.0

            return bs, F, p, sigma, L

        def soft_acceptance_scores(self, F: np.ndarray, L: np.ndarray, sigma: np.ndarray, p_c: float,
                                kappa: float = 1.0, epsilon: float = 1e-12):
            """
            Soft acceptance around p_c via a logistic weight:
            w_tau(b) = sigmoid( (L(b) - p_c) / tau(b) ), tau(b) = kappa * sigma(b)
            Returns:
            - C_soft: expected accepted coverage (weighted by w_tau)
            - C_max:  max_b F(b)*w_tau(b) (best soft-accepted coverage)
            - w_tau:  weights per b (for diagnostics)
            """
            if F.size == 0:
                return np.nan, np.nan, np.array([])

            tau = kappa * np.maximum(sigma, 1e-9)  # avoid division by zero
            w_tau = 1.0 / (1.0 + np.exp(-(L - p_c) / tau))
            Z = np.sum(w_tau) + epsilon
            C_soft = float(np.sum(w_tau * F) / Z)
            C_max = float(np.max(F * w_tau)) if F.size > 0 else np.nan
            return C_soft, C_max, w_tau

        def margin_and_coverage(self, L: np.ndarray, F: np.ndarray, p_c: float):
            """
            Margin M = max_b L(b) - p_c and coverage F* at argmax L.
            Negative M => below threshold by |M|.
            """
            if L.size == 0:
                return np.nan, np.nan
            j = int(np.argmax(L))
            M = float(L[j] - p_c)
            F_star = float(F[j])  # independent of p_c
            return M, F_star

        # ---------- 2D sweep: A & B for all p_c in p_c_list ----------

        def sweep_2d_soft_and_margin_for_pcs(
            self,
            net: "DirectedHomophilicNetwork",
            m_min: int, m_max: int, m_step: int,
            n0_min: int, n0_max: int, n0_step: int,
            node_type: str = 'b',
            a: int = 0,
            candidate_bs=None,
            b_min: int = None,
            b_max: int = None,
            n_b: int = 20,
            b_grid_type: str = 'linear',
            N_sims: int = 20,
            z: float = 1.0,
            p_c_list=(0.2,),
            kappa: float = 1.0,
            epsilon: float = 1e-12,
        ):
            """
            Compute 2D grids for each p_c in p_c_list:
            - C_soft (soft-accepted coverage)
            - C_max  (best soft-accepted coverage)
            - Margin M = p* - p_c
            Also returns a single Fstar_grid (coverage at argmax L), independent of p_c.
            """
            m_values = np.arange(m_min, m_max + 1, m_step, dtype=int)
            n0_values = np.arange(n0_min, n0_max + 1, n0_step, dtype=int)

            # p_c keyed containers:
            C_soft_grids = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
            C_max_grids  = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
            Margin_grids = {p_c: np.full((len(n0_values), len(m_values)), np.nan) for p_c in p_c_list}
            # F* does not depend on p_c
            Fstar_grid = np.full((len(n0_values), len(m_values)), np.nan)

            print(f"\n2D (soft+margin, multi p_c): m ∈ [{m_min}, {m_max}], n0 ∈ [{n0_min}, {n0_max}], p_c_list={p_c_list}, kappa={kappa}")

            total_points = len(n0_values) * len(m_values)
            done = 0

            for i, n0 in enumerate(n0_values):
                for j, m in enumerate(m_values):
                    done += 1
                    if n0 <= m:
                        continue

                    print(f"  Point {done}/{total_points}: (m={m}, n0={n0})")
                    net_temp = DirectedHomophilicNetwork(
                        n0=n0, n_nodes=net.n_nodes, m_edges=m,
                        h=net.h, f_a=net.f_a, mu_a=net.mu['a'], mu_b=net.mu['b']
                    )
                    net_temp.generate_network()

                    degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                    if degrees.size == 0:
                        continue

                    beta_theory = net_temp.get_beta_for_type(node_type)
                    scan_res = self.gof.scan_over_b(
                        net=net_temp,
                        node_type=node_type,
                        a=a,
                        beta_theory=beta_theory,
                        candidate_bs=candidate_bs,
                        N_sims=N_sims,
                        b_min=b_min,
                        b_max=b_max,
                        n_b=n_b,
                        b_grid_type=b_grid_type,
                    )

                    bs, F, p, sigma, L = self.build_series_for_scan(scan_res, degrees, a=a, z=z)
                    if bs.size == 0:
                        continue

                    # Compute F* once (independent of p_c)
                    if np.isnan(Fstar_grid[i, j]):
                        _, F_star = self.margin_and_coverage(L, F, p_c=0.0)  # p_c unused for F_star
                        Fstar_grid[i, j] = F_star

                    # Per p_c metrics
                    for p_c in p_c_list:
                        C_soft, C_max, _ = self.soft_acceptance_scores(F, L, sigma, p_c=p_c, kappa=kappa, epsilon=epsilon)
                        M, _Fstar_ = self.margin_and_coverage(L, F, p_c=p_c)

                        C_soft_grids[p_c][i, j] = C_soft
                        C_max_grids[p_c][i, j] = C_max
                        Margin_grids[p_c][i, j] = M

            return {
                'm_values': m_values,
                'n0_values': n0_values,
                'C_soft_grids': C_soft_grids,  # dict: p_c -> 2D array
                'C_max_grids': C_max_grids,    # dict: p_c -> 2D array
                'Margin_grids': Margin_grids,  # dict: p_c -> 2D array
                'Fstar_grid': Fstar_grid,      # 2D array
                'p_c_list': tuple(p_c_list),
                'kappa': kappa,
                'epsilon': epsilon,
                'z': z,
                'a': a,
                'node_type': node_type,
            }

        # ---------- Plotting: Hard acceptance (existing) moved here ----------

        def plot_hard_acceptance_contours_and_surfaces(self, grid_results: dict, metric: str = 'nodes', figsize_per_plot=(8, 6)):
            """
            Replicates the prior plot_2d_grid_results in a tidy home.
            metric ∈ {'nodes','edges'}
            Returns (fig_contour, fig_surface)
            """
            m_values = grid_results['m_values']
            n0_values = grid_results['n0_values']
            p_c_list = grid_results['p_c_list']
            z = grid_results['z']
            a = grid_results['a']
            node_type = grid_results['node_type']

            if metric == 'nodes':
                data_grids = grid_results['frac_nodes_grid']
                metric_label = 'Fraction of nodes kept'
                cbar_label = 'Nodes kept'
            else:
                data_grids = grid_results['frac_edges_grid']
                metric_label = 'Fraction of edges kept'
                cbar_label = 'Edges kept'

            M, N0 = np.meshgrid(m_values, n0_values)
            valid_mask = N0 > M

            # Contour
            fig_contour, ax = plt.subplots(1, len(p_c_list), figsize=(figsize_per_plot[0]*len(p_c_list), figsize_per_plot[1]))
            axes = [ax] if len(p_c_list) == 1 else ax

            for idx, p_c in enumerate(p_c_list):
                ax_i = axes[idx]
                data = data_grids[p_c].copy()
                data[~valid_mask] = np.nan

                levels = np.linspace(0, 1, 21)
                contourf = ax_i.contourf(M, N0, data, levels=levels, cmap='viridis', extend='both')
                contour_lines = ax_i.contour(M, N0, data, levels=levels[::4], colors='white', linewidths=0.5, alpha=0.4)
                ax_i.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
                ax_i.fill_between(m_values, m_values, n0_values[0], color='gray', alpha=0.3, label='Invalid (n₀ ≤ m)')

                cbar = fig_contour.colorbar(contourf, ax=ax_i)
                cbar.set_label(cbar_label, fontsize=10)
                ax_i.set_xlabel('m_edges', fontsize=11)
                ax_i.set_ylabel('n₀', fontsize=11)
                ax_i.set_title(f'p_c = {p_c}\n{metric_label} (type {node_type}, a={a}, z={z})', fontsize=11, fontweight='bold')
                ax_i.grid(True, alpha=0.3, linestyle='--')
                ax_i.legend(loc='upper left', fontsize=8)

            fig_contour.suptitle(f'2D Grid: {metric_label}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Surface (3D) — separate figure per p_c
            fig_surfaces = []
            for p_c in p_c_list:
                data = data_grids[p_c].copy()
                data[~valid_mask] = np.nan
                fig_3d = plt.figure(figsize=(figsize_per_plot[0], figsize_per_plot[1]))
                ax3 = fig_3d.add_subplot(111, projection='3d')
                surf = ax3.plot_surface(M, N0, data, cmap='viridis', alpha=0.9, edgecolor='none', antialiased=True)
                ax3.contour(M, N0, data, levels=10, offset=0, cmap='viridis', linewidths=1, alpha=0.5)
                ax3.set_xlabel('m_edges', fontsize=10)
                ax3.set_ylabel('n₀', fontsize=10)
                ax3.set_zlabel(cbar_label, fontsize=10)
                ax3.set_title(f'{metric_label} (p_c = {p_c}, type {node_type}, a={a}, z={z})', fontsize=11, fontweight='bold')
                ax3.set_zlim(0, 1)
                fig_3d.colorbar(surf, ax=ax3, shrink=0.6, aspect=14)
                ax3.view_init(elev=25, azim=45)
                plt.tight_layout()
                fig_surfaces.append(fig_3d)

            return fig_contour, fig_surfaces

        # ---------- Plotting: A) Soft acceptance (per p_c) ----------

        def plot_soft_acceptance_contour(self, grid_results_pcs: dict, p_c: float, which: str = 'C_soft', figsize=(8, 6)):
            """
            which ∈ {'C_soft','C_max'}; plots single contour for a given p_c.
            """
            m_values = grid_results_pcs['m_values']
            n0_values = grid_results_pcs['n0_values']
            M, N0 = np.meshgrid(m_values, n0_values)

            if which == 'C_soft':
                data = grid_results_pcs['C_soft_grids'][p_c].copy()
                title_core = 'Soft-accepted coverage C_soft'
            else:
                data = grid_results_pcs['C_max_grids'][p_c].copy()
                title_core = 'Best soft-accepted coverage C_max'

            valid_mask = N0 > M
            data[~valid_mask] = np.nan

            fig, ax = plt.subplots(figsize=figsize)
            levels = np.linspace(0, 1, 21)
            cs = ax.contourf(M, N0, data, levels=levels, cmap='viridis', extend='both')
            cl = ax.contour(M, N0, data, levels=levels[::4], colors='white', linewidths=0.6, alpha=0.5)
            ax.clabel(cl, inline=True, fontsize=8, fmt='%.2f')
            ax.fill_between(m_values, m_values, n0_values[0], color='gray', alpha=0.25, label='Invalid (n₀ ≤ m)')

            cbar = fig.colorbar(cs, ax=ax)
            cbar.set_label(which, fontsize=10)

            kappa = grid_results_pcs.get('kappa', None)
            z = grid_results_pcs.get('z', None)
            a = grid_results_pcs.get('a', None)
            node_type = grid_results_pcs.get('node_type', None)

            title = f'{title_core} (p_c={p_c}, type {node_type}, a={a}, κ={kappa}, z={z})'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('m_edges')
            ax.set_ylabel('n₀')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=8)
            plt.tight_layout()
            return fig

        def plot_soft_acceptance_surface(self, grid_results_pcs: dict, p_c: float, which: str = 'C_soft', figsize=(8, 6)):
            """
            3D surface for C_soft or C_max at a given p_c.
            """

            m_values = grid_results_pcs['m_values']
            n0_values = grid_results_pcs['n0_values']
            M, N0 = np.meshgrid(m_values, n0_values)

            if which == 'C_soft':
                data = grid_results_pcs['C_soft_grids'][p_c].copy()
                title_core = 'C_soft (soft-accepted coverage)'
            else:
                data = grid_results_pcs['C_max_grids'][p_c].copy()
                title_core = 'C_max (best soft-accepted coverage)'

            valid_mask = N0 > M
            data[~valid_mask] = np.nan

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(M, N0, data, cmap='viridis', alpha=0.95, edgecolor='none', antialiased=True)
            ax.set_xlabel('m_edges')
            ax.set_ylabel('n₀')
            ax.set_zlabel(which)
            ax.set_zlim(0, 1)
            ax.view_init(elev=25, azim=45)
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=14)

            kappa = grid_results_pcs.get('kappa', None)
            z = grid_results_pcs.get('z', None)
            a = grid_results_pcs.get('a', None)
            node_type = grid_results_pcs.get('node_type', None)
            ax.set_title(f'{title_core} (p_c={p_c}, type {node_type}, a={a}, κ={kappa}, z={z})', fontsize=11, fontweight='bold')
            plt.tight_layout()
            return fig

        # ---------- Plotting: B) Margin (per p_c), overlay F* on contour ----------

        def plot_margin_contour_with_coverage(self, grid_results_pcs: dict, p_c: float, F_levels=(0.25, 0.5, 0.75), figsize=(8, 6)):
            """
            Diverging color for Margin M = p* - p_c, with F* contours, at a given p_c.
            """
            m_values = grid_results_pcs['m_values']
            n0_values = grid_results_pcs['n0_values']
            M, N0 = np.meshgrid(m_values, n0_values)

            data_M = grid_results_pcs['Margin_grids'][p_c].copy()
            data_F = grid_results_pcs['Fstar_grid'].copy()
            valid_mask = N0 > M
            data_M[~valid_mask] = np.nan
            data_F[~valid_mask] = np.nan

            fig, ax = plt.subplots(figsize=figsize)
            max_abs = np.nanmax(np.abs(data_M))
            if not np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1.0
            levels = np.linspace(-max_abs, max_abs, 21)
            cs = ax.contourf(M, N0, data_M, levels=levels, cmap='coolwarm', extend='both')
            cbar = fig.colorbar(cs, ax=ax)
            cbar.set_label('Margin M = p* - p_c', fontsize=10)

            # F* contours
            cl = ax.contour(M, N0, data_F, levels=F_levels, colors='black', linewidths=1.0, linestyles='--')
            ax.clabel(cl, inline=True, fontsize=8, fmt='F* = %.2f')

            kappa = grid_results_pcs.get('kappa', None)
            z = grid_results_pcs.get('z', None)
            a = grid_results_pcs.get('a', None)
            node_type = grid_results_pcs.get('node_type', None)

            ax.fill_between(m_values, m_values, n0_values[0], color='gray', alpha=0.25, label='Invalid (n₀ ≤ m)')
            ax.set_title(f'Margin (p_c={p_c}, type {node_type}, a={a}, z={z}) with F* contours', fontsize=12, fontweight='bold')
            ax.set_xlabel('m_edges')
            ax.set_ylabel('n₀')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=8)
            plt.tight_layout()
            return fig

        def plot_margin_surface(self, grid_results_pcs: dict, p_c: float, figsize=(8, 6)):
            """
            3D surface for Margin M = p* - p_c at a given p_c.
            """
            m_values = grid_results_pcs['m_values']
            n0_values = grid_results_pcs['n0_values']
            M, N0 = np.meshgrid(m_values, n0_values)

            data_M = grid_results_pcs['Margin_grids'][p_c].copy()
            valid_mask = N0 > M
            data_M[~valid_mask] = np.nan

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(M, N0, data_M, cmap='coolwarm', alpha=0.95, edgecolor='none', antialiased=True)
            ax.set_xlabel('m_edges')
            ax.set_ylabel('n₀')
            ax.set_zlabel('Margin M')
            ax.view_init(elev=25, azim=45)
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=14)

            z = grid_results_pcs.get('z', None)
            a = grid_results_pcs.get('a', None)
            node_type = grid_results_pcs.get('node_type', None)
            ax.set_title(f'Margin M = p* - p_c (p_c={p_c}, type {node_type}, a={a}, z={z})', fontsize=11, fontweight='bold')
            plt.tight_layout()
            return fig

class NetworkPlotting:
    """
    Plotting utilities for DirectedHomophilicNetwork.
    Methods take a `DirectedHomophilicNetwork` instance `net`.
    """

    def logarithmic_binning(self,degrees: np.ndarray,bin_factor: float = 1.01) -> Tuple[np.ndarray, np.ndarray]:
        """Create logarithmic bins with k=0 always in its own bin."""
        if len(degrees) == 0:
            return np.array([]), np.array([])  # prevents crash out if no nodes of a given type exist

        degrees = np.array(degrees)
        max_degree, n_total = np.max(degrees), len(degrees)

        # Handle k=0 separately
        count_zero = np.sum(degrees == 0)
        bin_centers = [0.0] if count_zero > 0 else []
        probabilities = [count_zero / n_total] if count_zero > 0 else []

        # Create logarithmic sequence for k >= 1
        if np.any(degrees > 0):
            bins = [1]
            current = 1 * bin_factor
            while current <= max_degree:
                bins.append(int(current))
                current *= bin_factor
            bins.append(int(max_degree) + 1)
            bins = sorted(set(bins))

            # Compute bin statistics for k > 0: turn the logarithmic sequence into bins
            # and count how many degrees fall into each bin
            for i in range(len(bins) - 1):
                kmin, kmax = bins[i], bins[i + 1] - 1
                count = np.sum((degrees >= kmin) & (degrees <= kmax))

                if count > 0:
                    center = np.sqrt(kmin * kmax)  # Geometric mean (always kmin > 0)
                    bin_centers.append(center)
                    probabilities.append(count / n_total)

        return np.array(bin_centers), np.array(probabilities)

    def plot_degree_distributions(self,net: DirectedHomophilicNetwork,figsize: Tuple = (15, 6),discretisations: int = 10**5,):
        """
        Plot in-degree distributions with theoretical curves. If discretisations=0,
        theoretical curve is only computed at integer k.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
            in_degrees = net._get_degrees(node_type)
            if len(in_degrees) == 0:
                continue

            ax = axes[idx]

            # Empirical data
            bin_centers, probs = self.logarithmic_binning(in_degrees)
            ax.scatter(
                bin_centers,
                probs,
                s=50,
                alpha=0.7,
                color=color,
                edgecolors='black',
                linewidths=0.5,
                label='Simulation',
                zorder=3,
            )

            ax.axvline(
                x=0,
                color='black',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                zorder=4,
            )

            # Theoretical curve
            k_max = int(np.max(in_degrees))
            if discretisations == 0:
                k_range = np.arange(0, k_max + 1)
            else:
                k_range = np.concatenate([[0], np.linspace(0.01, k_max, discretisations)])
            theo_probs = net.theoretical_distribution(k_range, node_type)
            mask = theo_probs > 0

            ax.plot(
                k_range[mask],
                theo_probs[mask],
                '-',
                linewidth=2.5,
                color='dark' + color,
                alpha=0.85,
                label='Theory',
                zorder=2,
            )

            # Formatting
            ax.set_xscale('symlog', linthresh=0.1)
            ax.set_yscale('log')
            ax.set_xlim(left=-0.05)
            ax.set_xlabel(r'In-degree $k^{\mathrm{(in)}}$', fontsize=13)
            ax.set_ylabel(r'Probability $p(k^{\mathrm{(in)}})$', fontsize=13)
            ax.set_title(
                f'Type "{node_type}" (n={len(in_degrees)})',
                fontsize=13,
                fontweight='bold',
            )
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)

        fig.suptitle(
            f'Directed Homophilic Network: N={net.n0 + net.n_nodes:,}, '
            f'm={net.m_edges}, h={net.h}, f_a={net.f_a}',
            fontsize=14,
            fontweight='bold',
        )
        plt.tight_layout()
        return fig

    def plot_in_edge_asymptotes(self,net: DirectedHomophilicNetwork,figsize: Tuple = (10, 6),):
        """Plot mean in-edge density with asymptotic fits."""
        times = np.array([d['t'] for d in net.edge_evolution])
        mean_deg_a = np.array([d['in_edges_a'] / d['t'] for d in net.edge_evolution])
        mean_deg_b = np.array([d['in_edges_b'] / d['t'] for d in net.edge_evolution])

        fig, ax = plt.subplots(figsize=figsize)

        for mean_deg, type_name, color in [
            (mean_deg_a, 'a', 'red'),
            (mean_deg_b, 'b', 'blue'),
        ]:
            ax.plot(times, mean_deg, label=f"Type '{type_name}' (data)", color=color)
            asymptote = net.g_a if type_name == 'a' else net.g_b
            ax.axhline(
                asymptote,
                linestyle='--',
                alpha=0.7,
                color='dark' + color,
                label=f"Type '{type_name}' asymptote = {asymptote:.3f}",
            )

        ax.set_xlabel("t (number of nodes)")
        ax.set_ylabel("Mean in-degree")
        ax.set_title("Asymptotic In-Edge Density")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_A_values(self, net: DirectedHomophilicNetwork, max_k: int = 25):
        """Plot normalization constant A(k) for both types."""
        k_values = np.arange(0, max_k + 1)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
            params = net._get_params(node_type)

            # Compute A(k) by brute force
            A_values = []
            for k in k_values:
                product = (
                    np.prod(
                        [
                            (params['alpha'] + i)
                            / (params['alpha'] + params['gamma'] + i)
                            for i in range(k)
                        ]
                    )
                    if k > 0
                    else 1.0
                )
                gamma_ratio = gamma_func(
                    k + params['alpha'] + params['gamma']
                ) / gamma_func(k + params['alpha'])
                A_values.append(params['p0'] * product * gamma_ratio)

            ax = axes[idx]
            ax.plot(
                k_values,
                A_values,
                'o-',
                color=color,
                linewidth=2,
                markersize=4,
                alpha=0.7,
                label='A(k) computed',
            )
            ax.axhline(
                params['A'],
                linestyle='--',
                color='black',
                linewidth=2,
                alpha=0.7,
                label=f"b₀·Γ(α+γ)/Γ(α) = {params['A']:.2f}",
            )

            ax.set_xlabel('k', fontsize=13)
            ax.set_ylabel('A(k)', fontsize=13)
            ax.set_title(
                f"Type '{node_type}' Normalization",
                fontsize=13,
                fontweight='bold',
            )
            ax.set_xscale('log')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_degree_distributions_discrete(self,net: DirectedHomophilicNetwork,figsize: Tuple = (15, 6),max_k_display: int = None,):
        """
        Plot using discrete integer probabilities - no binning.
        This shows the true empirical PMF at each integer k value.
        LINEAR SCALE VERSION - raw data.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
            in_degrees = net._get_degrees(node_type)
            if len(in_degrees) == 0:
                continue

            ax = axes[idx]

            # Empirical PMF at integer values
            unique_k, counts = np.unique(in_degrees, return_counts=True)
            empirical_pmf = counts / len(in_degrees)

            ax.scatter(
                unique_k,
                empirical_pmf,
                s=1,
                alpha=0.7,
                color=color,
                edgecolors='black',
                linewidths=1,
                label='Simulation',
                zorder=3,
            )

            # Theoretical curve at integer values
            k_max = int(np.max(in_degrees)) if max_k_display is None else max_k_display
            k_range = np.arange(0, k_max + 1)
            theo_probs = net.theoretical_distribution(k_range, node_type)

            ax.plot(
                k_range,
                theo_probs,
                '-',
                linewidth=1,
                color='dark' + color,
                alpha=1,
                label='Theory',
                zorder=2,
            )

            # Formatting - LINEAR AXES
            ax.set_xlabel(r'In-degree ${k^{\mathrm{(in)}}}$', fontsize=13)
            ax.set_ylabel(r'Probability ${p(k^{\mathrm{(in)}})}$', fontsize=13)
            ax.set_title(
                f'Type "{node_type}" (n={len(in_degrees)}) - LINEAR SCALE',
                fontsize=13,
                fontweight='bold',
            )
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        fig.suptitle(
            f'Directed Homophilic Network: N={net.n0 + net.n_nodes:,}, '
            f'm={net.m_edges}, h={net.h}, f_a={net.f_a}',
            fontsize=14,
            fontweight='bold',
        )
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

if __name__ == "__main__":

    config = dict(
        network=dict(
            n0=30, n_nodes=5000, m_edges=3,
            h=0.2, f_a=0.2, mu_a=1, mu_b=5, 
            seed=None,
        ),
        sweep_m=dict(
            m_min=1, m_max= 25, m_step=1,
            node_type='b', a=0,
            p_c_list=[0.1, 0.4],
            N_sims=30, b_grid_type='linear',
            n_b=50, b_min=None, b_max=None,
        ),
        sweep_n0=dict(
            n0_min=10, n0_max=150, n0_step=10,
            node_type='b', a=0,
            p_c_list=[0.1, 0.4],
            N_sims= 30, b_grid_type='linear',
            n_b=50, b_min=None, b_max=None,
        ),
        grid_2d=dict(
            m_min=1, m_max=25, m_step=5,
            n0_min=10, n0_max=150, n0_step=50,
            node_type='b', a=0,
            p_c_list=[0.1, 0.4],
            N_sims=5, b_grid_type='linear', n_b=10,
            b_min=None, b_max=None, z=1.0, 
            kappa=3, epsilon=1e-12,
        ),
        plots=dict(
            log_binned=True, discrete_linear=True,
            asymptotes=True, A_const=True,
            sweep_m_edges_csn=False, sweep_n0_csn= False,
            csn_p_vs_b_m=3, csn_p_vs_b_n0=3,
            grid_2d_sweep=True, 
        ))
    
    fm = FileManager(config)
    net = DirectedHomophilicNetwork(**config["network"])
    gof = GoFDiagnostics()
    plotting = NetworkPlotting()
    stats = NetworkStatistics()

    start = time.time()
    net.generate_network()
    print(f"Network generated in {time.time() - start:.2f}s")

    stats.print_statistics(net)

    plots = config["plots"]

    if plots["log_binned"]:
        fig = plotting.plot_degree_distributions(net)
        fm.save_fig(fig, "degree_dist_log_binned")

    if plots["discrete_linear"]:
        fig = plotting.plot_degree_distributions_discrete(net)
        fm.save_fig(fig, "degree_dist_discrete")

    if plots["asymptotes"]:
        fig = plotting.plot_in_edge_asymptotes(net)
        fm.save_fig(fig, "asymptotes")

    if plots["A_const"]:
        fig = plotting.plot_A_values(net)
        fm.save_fig(fig, "A_const")

    if plots["sweep_m_edges_csn"]:
        results_m = gof.csn_sweep_m_edges(net, **config["sweep_m"])
        fm.save_fig(results_m["fig"], "sweep_m_edges_csn")

        num_p_vs_b_m = plots.get("csn_p_vs_b_m", 0)
        if num_p_vs_b_m > 0:
            gof.csn_diagnostic_plots_over_sweep(
                sweep_results=results_m,
                p_c_list=config["sweep_m"]["p_c_list"],
                num_p_vs_b=num_p_vs_b_m,
                fm=fm,
                node_type=config["sweep_m"]["node_type"],
                sweep_label='m_edges',
                z=1.0,
                a=config["sweep_m"]["a"],
            )

    if plots["sweep_n0_csn"]:
        results_n0 = gof.csn_sweep_n0(net, **config["sweep_n0"])
        fm.save_fig(results_n0["fig"], "sweep_n0_csn")

        num_p_vs_b_n0 = plots.get("csn_p_vs_b_n0", 0)
        if num_p_vs_b_n0 > 0:
            gof.csn_diagnostic_plots_over_sweep(
                sweep_results=results_n0,
                p_c_list=config["sweep_n0"]["p_c_list"],
                num_p_vs_b=num_p_vs_b_n0,
                fm=fm,
                node_type=config["sweep_n0"]["node_type"],
                sweep_label='n0',
                z=1.0,
                a=config["sweep_n0"]["a"],
            )

    if plots["grid_2d_sweep"]:
        # Filter only the kwargs that csn_sweep_2d_grid accepts
        _allowed_keys_grid2d = {
            "m_min", "m_max", "m_step",
            "n0_min", "n0_max", "n0_step",
            "node_type", "a",
            "p_c_list",
            "N_sims", "b_grid_type", "n_b",
            "b_min", "b_max",
            "z",
        }
        _grid2d_args = {k: v for k, v in config["grid_2d"].items() if k in _allowed_keys_grid2d}

        grid_results = gof.csn_sweep_2d_grid(net, **_grid2d_args)

        # Tidy plotting via the new class
        v2d = gof.CSN2DVis(gof)

        # Hard-acceptance plots (nodes and edges)
        fig_cont_nodes, fig_surfaces_nodes = v2d.plot_hard_acceptance_contours_and_surfaces(grid_results, metric='nodes')
        fm.save_fig(fig_cont_nodes, "grid2d_hard_nodes_contour")
        for idx, fig3 in enumerate(fig_surfaces_nodes):
            fm.save_fig(fig3, f"grid2d_hard_nodes_surface_pc{config['grid_2d']['p_c_list'][idx]}")

        fig_cont_edges, fig_surfaces_edges = v2d.plot_hard_acceptance_contours_and_surfaces(grid_results, metric='edges')
        fm.save_fig(fig_cont_edges, "grid2d_hard_edges_contour")
        for idx, fig3 in enumerate(fig_surfaces_edges):
            fm.save_fig(fig3, f"grid2d_hard_edges_surface_pc{config['grid_2d']['p_c_list'][idx]}")

        # A & B: soft acceptance + margin (this call still uses kappa/epsilon)
        grid_soft_margin = v2d.sweep_2d_soft_and_margin_for_pcs(
            net=net,
            m_min=config["grid_2d"]["m_min"],
            m_max=config["grid_2d"]["m_max"],
            m_step=config["grid_2d"]["m_step"],
            n0_min=config["grid_2d"]["n0_min"],
            n0_max=config["grid_2d"]["n0_max"],
            n0_step=config["grid_2d"]["n0_step"],
            node_type=config["grid_2d"]["node_type"],
            a=config["grid_2d"]["a"],
            candidate_bs=None,
            b_min=config["grid_2d"]["b_min"],
            b_max=config["grid_2d"]["b_max"],
            n_b=config["grid_2d"]["n_b"],
            b_grid_type=config["grid_2d"]["b_grid_type"],
            N_sims=config["grid_2d"]["N_sims"],
            z=config["grid_2d"]["z"],
            p_c_list=config["grid_2d"]["p_c_list"],
            kappa=config["grid_2d"]["kappa"],       # stays
            epsilon=config["grid_2d"]["epsilon"],   # stays
        )

        # Plot per p_c for soft-acceptance (C_soft & C_max) and margin
        for p_c in config["grid_2d"]["p_c_list"]:
            fig_csoft = v2d.plot_soft_acceptance_contour(grid_soft_margin, p_c=p_c, which='C_soft')
            fm.save_fig(fig_csoft, f"grid2d_soft_Csoft_contour_pc{p_c}")
            fig_csoft3 = v2d.plot_soft_acceptance_surface(grid_soft_margin, p_c=p_c, which='C_soft')
            fm.save_fig(fig_csoft3, f"grid2d_soft_Csoft_surface_pc{p_c}")

            fig_cmax = v2d.plot_soft_acceptance_contour(grid_soft_margin, p_c=p_c, which='C_max')
            fm.save_fig(fig_cmax, f"grid2d_soft_Cmax_contour_pc{p_c}")
            fig_cmax3 = v2d.plot_soft_acceptance_surface(grid_soft_margin, p_c=p_c, which='C_max')
            fm.save_fig(fig_cmax3, f"grid2d_soft_Cmax_surface_pc{p_c}")

            fig_margin = v2d.plot_margin_contour_with_coverage(grid_soft_margin, p_c=p_c, F_levels=(0.25, 0.5, 0.75))
            fm.save_fig(fig_margin, f"grid2d_margin_contour_pc{p_c}")
            fig_margin3 = v2d.plot_margin_surface(grid_soft_margin, p_c=p_c)
            fm.save_fig(fig_margin3, f"grid2d_margin_surface_pc{p_c}")

        fm.finalize_metadata()
        print(f"\nAll outputs saved to: {fm.path()}")