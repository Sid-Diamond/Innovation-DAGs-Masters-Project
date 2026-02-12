import numpy as np
import networkx as nx
import matplotlib.pyplot as plt  
from scipy.special import gamma as gamma_func, beta as beta_func
from typing import Dict, Tuple, List
import time
from scipy.optimize import minimize
from scipy.special import beta as beta_func


class DirectedHomophilicNetwork:
    """Optimized directed network with homophilic preferential attachment."""

    def __init__(
        self,
        n0: int,
        n_nodes: int,
        m_edges: int,
        h: float,
        f_a: float,
        mu_a: float,
        mu_b: float,
        seed: int = None
    ):
        # Network parameters
        self.n0, self.n_nodes, self.m_edges = n0, n_nodes, m_edges
        self.h, self.f_a, self.f_b = h, f_a, 1 - f_a
        self.mu = {'a': mu_a, 'b': mu_b}
        self.seed = seed

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
        self.g_a, self.g_b, self.g_b_empirical, self.Z_factor, self.Z_tilde = (
            None,
            None,
            None,
            None,
            None,
        )

        # Cached state for speed
        self.in_degrees = None
        self.in_edges_a_count = 0
        self.in_edges_b_count = 0

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
        self.g_b_empirical = g_b_empirical  # Store for comparison only
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

        return np.random.choice(
            n_nodes_so_far, size=self.m_edges, p=probs, replace=False
        )

    def generate_network(self):
        """Generate network with cached state."""
        total_nodes = self.n0 + self.n_nodes

        # Pre-allocate arrays
        self.node_types = np.empty(total_nodes, dtype=np.int8)  # 0 for 'a', 1 for 'b'
        self.in_degrees = np.zeros(total_nodes, dtype=np.int32)
        self.graph = nx.DiGraph()

        # Initialize nodes
        for i in range(self.n0):
            self.node_types[i] = 0 if self.assign_node_type() == 'a' else 1
            self.graph.add_node(i)

        # Initial random edges
        for source in range(self.n0):
            targets = np.random.choice(
                [t for t in range(self.n0) if t != source],
                size=self.m_edges,
                replace=False,
            )
            # select m targets without self-loops without seleccting same target multiple times. Node type irrelevant here.
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
            self.graph.add_node(new_node)

            targets = self.homophilic_preferential_attachment(new_node)
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
                # Stores in edge counts for every y = n0 +100t and at the final datapoint for a non memory
                # intensive asymptotic g value calculation

        self._fit_asymptotes()

    def _fit_asymptotes(self, fraction: float = 0.05):
        """Fit asymptotic g values from evolution data."""
        mean_deg_a = np.array(
            [d['in_edges_a'] / d['t'] for d in self.edge_evolution]
        )
        mean_deg_b = np.array(
            [d['in_edges_b'] / d['t'] for d in self.edge_evolution]
        )

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
        Core PMF shell: p(k | beta) with beta = (p0, alpha, gamma).

        This is the only place where the Yule–Simon–type form is implemented.
        """
        p0, alpha, gamma = beta
        k = np.atleast_1d(k)
        result = np.zeros_like(k, dtype=float)

        zero_mask = (k == 0)
        pos_mask = (k > 0)

        # k = 0
        if np.any(zero_mask):
            result[zero_mask] = p0

        # k > 0
        if np.any(pos_mask):
            k_pos = k[pos_mask]
            result[pos_mask] = (
                p0
                * beta_func(k_pos, alpha + gamma)
                / beta_func(k_pos, alpha)
            )

        return result

    def theoretical_distribution(self, k, node_type: str):
        """
        Theoretical in-degree distribution with analytic continuation.

        Implemented via the generic shell pmf_from_beta using the
        network's current (p0, alpha, gamma) for this node_type.
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

        support = np.arange(kmin, kmax + 1, dtype=int)
        pmf = self.pmf_from_beta(support, beta).astype(float)

        Z = pmf.sum()
        if Z <= 0:
            # Degenerate case: CDF is 0 until kmax, then 1
            cdf_support = np.ones_like(support, dtype=float)
        else:
            pmf /= Z
            cdf_support = np.cumsum(pmf)

        F = np.zeros_like(k_int, dtype=float)
        for i, ki in enumerate(k_int):
            if ki < kmin:
                F[i] = 0.0
            elif ki >= kmax:
                F[i] = 1.0
            else:
                F[i] = cdf_support[ki - kmin]

        return F.item() if F.size == 1 else F

    def _get_degrees(self, node_type: str) -> List[int]:
        """Get in-degrees for nodes of specified type."""
        type_val = 0 if node_type == 'a' else 1
        return [
            self.graph.in_degree(n)
            for n in self.graph.nodes()
            if self.node_types[n] == type_val
        ]
    
class GoFDiagnostics:
    """
    Goodness-of-Fit and diagnostic methods for DirectedHomophilicNetwork.
    Holds only cosmetic separation of concerns; methods still expect to
    receive a `DirectedHomophilicNetwork` instance as `net`.
    """

    def theoretical_cdf_discrete(
        self,
        net: DirectedHomophilicNetwork,
        k,
        node_type: str,
        kmin: int,
        kmax: int,
        beta=None,
    ):
        """
        Discrete CDF F(k | beta, kmin <= K <= kmax) on integers kmin,...,kmax.

        Parameters
        ----------
        net : DirectedHomophilicNetwork
            Provides pmf_from_beta and get_beta_for_type.
        k : scalar or array-like
            Points at which to evaluate the CDF.
        node_type : str
            'a' or 'b'; used only if beta is None.
        kmin, kmax : int
            Truncation window [kmin, kmax].
        beta : (p0, alpha, gamma) or None
            If provided, use this beta.
            If None, use the network's beta for this node_type.
        """
        k = np.atleast_1d(k)

        if beta is None:
            beta = net.get_beta_for_type(node_type)

        return net.cdf_from_beta_truncated(k, beta, kmin, kmax)

    def empirical_cdf_integers(self, data):
        """
        Compute the empirical CDF for integer data only (vectorized).
        """
        data = np.array(data, dtype=int)
        n = len(data)
        if n == 0:
            return np.array([]), np.array([])

        unique_vals = np.arange(data.min(), data.max() + 1)
        # Vectorized: use searchsorted to count how many elements <= each unique value
        sorted_data = np.sort(data)
        counts = np.searchsorted(sorted_data, unique_vals, side='right')
        cdf_vals = counts / n

        return unique_vals, cdf_vals

    def csn_distance(self,net: DirectedHomophilicNetwork,data,a: int,b: int, beta, node_type: str,) -> float:
        """
        CSN/AD-style distance on a truncated window [a,b]:

            D(data; beta; [a,b]) = max_{n = a,...,b}
                | (N_n / N_a) - S(n; beta; [a,b]) |
                / sqrt( S(n; beta; [a,b]) * (1 - S(n; beta; [a,b])) )

        where:
          - data is already truncated so that a <= x <= b
          - N_a = len(data)
          - N_n = # of data points with x >= n
          - S(n; beta; [a,b]) is the theoretical CDF conditional on [a,b]
            **under the parameter vector beta = (p0, alpha, gamma)**.
        """

        # By assumption, all data lie in [a,b]
        N_a = data.size

        # Support n = a,..., b
        n_vals = np.arange(a, b + 1, dtype=int)

        # Sort data once
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

    def _fit_beta_mle(self, net: DirectedHomophilicNetwork, data, a: int, b: int, node_type: str, beta_init,):
        data = np.asarray(data, dtype=int)
        if data.size == 0:
            params0 = net._get_params(node_type)
            beta_mle = np.array([params0['p0'], params0['alpha'], params0['gamma']], dtype=float)
            return beta_mle, True

        params0 = net._get_params(node_type)
        if beta_init is None:
            p0_0 = float(params0['p0']); alpha0 = float(params0['alpha']); gamma0 = float(params0['gamma'])
        else:
            beta_init_arr = np.asarray(beta_init, dtype=float)
            p0_0, alpha0, gamma0 = beta_init_arr

        k_support = np.arange(a, b + 1, dtype=int)

        unique_k, counts = np.unique(data, return_counts=True)
        total = counts.sum()
        emp_pmf = counts / total

        def truncated_pmf(p0: float, alpha: float, gamma: float) -> np.ndarray:
            pmf = np.zeros_like(k_support, dtype=float)
            zero_mask = (k_support == 0)
            if np.any(zero_mask):
                pmf[zero_mask] = p0
            pos_mask = (k_support > 0)
            if np.any(pos_mask):
                k_pos = k_support[pos_mask]
                pmf[pos_mask] = p0 * beta_func(k_pos, alpha + gamma) / beta_func(k_pos, alpha)
            Z_trunc = pmf.sum()
            if Z_trunc <= 0:
                return np.full_like(k_support, 1e-300, dtype=float)
            pmf /= Z_trunc
            pmf = np.clip(pmf, 1e-300, 1.0)
            return pmf

        def objective(theta: np.ndarray) -> float:
            p0, alpha, gamma = theta
            model_pmf_full = truncated_pmf(p0, alpha, gamma)
            model_probs, emp_probs = [], []
            for k, p_emp in zip(unique_k, emp_pmf):
                if a <= k <= b:
                    idx = np.where(k_support == k)[0]
                    if idx.size == 0:
                        continue
                    model_probs.append(model_pmf_full[idx[0]])
                    emp_probs.append(p_emp)
            if not model_probs:
                return 1e6
            model_probs = np.asarray(model_probs, dtype=float)
            emp_probs = np.asarray(emp_probs, dtype=float)
            log_model = np.log(model_probs)
            log_emp = np.log(emp_probs)
            return np.sum((log_model - log_emp) ** 2)

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

    def mc_on_window(self, net: DirectedHomophilicNetwork, node_type: str, a: int, b: int, beta_theory, N_sims: int,):
        D_theory_list = []
        D_mle_list = []
        beta_mle_list = []
        fallback_flags = []

        beta_theory_arr = np.asarray(beta_theory) if beta_theory is not None else None

        for s in range(N_sims):
            net_sim = DirectedHomophilicNetwork(net.n0, net.n_nodes, net.m_edges, net.h, net.f_a, net.mu['a'], net.mu['b'], seed=None)
            net_sim.generate_network()

            degrees = np.array(net_sim._get_degrees(node_type), dtype=int)
            data_s = degrees[(degrees >= a) & (degrees <= b)]

            if data_s.size == 0:
                D_theory_list.append(0.0)
                D_mle_list.append(0.0)
                beta_mle_list.append(beta_theory_arr if beta_theory_arr is not None else np.array([np.nan, np.nan, np.nan]))
                fallback_flags.append(True)
                continue

            D_theory_s = self.csn_distance(net_sim, data_s, a, b, beta_theory, node_type)
            beta_mle_s, used_fallback_s = self._fit_beta_mle(net_sim, data_s, a, b, node_type, beta_init=None)
            D_mle_s = self.csn_distance(net_sim, data_s, a, b, beta_mle_s, node_type)

            D_theory_list.append(D_theory_s)
            D_mle_list.append(D_mle_s)
            beta_mle_list.append(beta_mle_s)
            fallback_flags.append(used_fallback_s)

        D_theory_arr = np.array(D_theory_list, dtype=float)
        D_mle_arr = np.array(D_mle_list, dtype=float)
        fallback_flags = np.asarray(fallback_flags, dtype=bool)

        greater_mask = D_mle_arr > D_theory_arr
        p = float(np.mean(greater_mask.astype(float)))
        sigma_p = float(np.sqrt(p * (1.0 - p) / max(N_sims, 1)))

        try:
            beta_mle_arr = np.asarray(beta_mle_list, dtype=float)
        except Exception:
            beta_mle_arr = np.array(beta_mle_list, dtype=object)

        frac_moved = float(np.mean(~fallback_flags)) if fallback_flags.size > 0 else np.nan

        beta_mle_mean = None
        beta_mle_std = None
        if isinstance(beta_mle_arr, np.ndarray) and beta_mle_arr.ndim == 2 and beta_mle_arr.shape[0] == fallback_flags.size:
            mask_non_fallback = ~fallback_flags
            if np.any(mask_non_fallback):
                beta_mle_non_fallback = beta_mle_arr[mask_non_fallback]
                beta_mle_mean = np.nanmean(beta_mle_non_fallback, axis=0)
                beta_mle_std = np.nanstd(beta_mle_non_fallback, axis=0)

        percent_diff_mean_vs_theory = None
        if beta_theory_arr is not None and beta_mle_mean is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                percent_diff_mean_vs_theory = 100.0 * ((beta_mle_mean - beta_theory_arr) / beta_theory_arr)
                percent_diff_mean_vs_theory = np.where(beta_theory_arr == 0.0, np.nan, percent_diff_mean_vs_theory)

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
    
    def scan_over_b(self,net: DirectedHomophilicNetwork,node_type: str,a: int,beta_theory,candidate_bs=None,N_sims: int = 20,b_min: int = None,b_max: int = None,n_b: int = 20,b_grid_type: str = 'linear',):
        bs = self._build_b_grid(net=net,node_type=node_type,a=a,candidate_bs=candidate_bs,b_min=b_min,b_max=b_max,n_b=n_b,b_grid_type=b_grid_type,)

        windows_results = []

        if bs is None or len(bs) == 0:
            # no valid b range; return empty windows
            return {
                'a': a,
                'beta_theory': beta_theory,
                'windows': [],
                'b_grid': None,
            }
    
        print(f"    b-grid (candidate b_j): {bs}")

        for b in bs:
            if b <= a:
                raise ValueError(f"Each b_j must satisfy b_j > a. Got a={a}, b_j={b}.")

            window_result = self.mc_on_window(net=net,node_type=node_type,a=a,b=int(b),beta_theory=beta_theory,N_sims=N_sims,)

            window_result_with_b = dict(window_result)
            window_result_with_b['b'] = int(b)
            windows_results.append(window_result_with_b)

        result = {
            'a': a,
            'beta_theory': beta_theory,
            'windows': windows_results,
            'b_grid': np.array(bs, dtype=int),
        }
        return result

    def scan_until_threshold(
        self,
        windows,
        a: int,
        p_c: float,
    ):
        """
        Given a list of window results (as produced in scan_over_b['windows']),
        and a threshold p_c, find the largest b where p >= p_c.

        windows: list of dicts, each with keys 'b', 'p', 'sigma_p', etc.
        """
        if not windows:
            return {
                'a': a,
                'p_c': float(p_c),
                'windows_evaluated': [],
                'largest_window': None,
            }

        # sort windows by b ascending, then traverse from largest to smallest
        windows_sorted = sorted(windows, key=lambda w: w['b'])
        windows_evaluated = []
        largest_window_info = None

        for j_rev, w in enumerate(reversed(windows_sorted)):
            b = int(w['b'])
            p_val = float(w['p'])
            sigma_p = float(w['sigma_p'])
            windows_evaluated.append(w)

            if p_val >= p_c:
                largest_window_info = {
                    'a': a,
                    'b': b,
                    'p': p_val,
                    'sigma_p': sigma_p,
                    'index': j_rev,
                }
                break

        return {
            'a': a,
            'p_c': float(p_c),
            'windows_evaluated': windows_evaluated,
            'largest_window': largest_window_info,
        }

    def select_largest_window_for_pcs(
        self,
        windows,
        a: int,
        p_c_list,
    ):
        """
        For a given list of windows (from scan_over_b['windows']) and
        a list of p_c values, return a dict:
        {
            p_c_value: scan_until_threshold_result_for_that_p_c,...
        }
        """
        results = {}
        for p_c in p_c_list:
            res = self.scan_until_threshold(windows=windows, a=a, p_c=p_c)
            results[p_c] = res
        return results

    def csn_sweep_m_edges(
        self,
        net: DirectedHomophilicNetwork,
        m_min: int,
        m_max: int,
        m_step: int = 1,
        node_type: str = 'b',
        a: int = 0,
        candidate_bs=None,
        b_min: int = None,
        b_max: int = None,
        n_b: int = 20,
        b_grid_type: str = 'linear',
        N_sims: int = 20,
        p_c_list=(0.1, 0.2, 0.4, 0.6),
        figsize=(12, 6),
    ):
        # global style
        plt.rcParams['font.family'] = 'Times New Roman'

        m_values = np.arange(m_min, m_max + 1, m_step, dtype=int)
        windows_info = []

        # per p_c, store frac_nodes_kept and frac_edges_kept and no-window flag
        frac_nodes_kept = {p_c: [] for p_c in p_c_list}
        frac_edges_kept = {p_c: [] for p_c in p_c_list}
        no_window = {p_c: [] for p_c in p_c_list}

        print(f"\nCSN sweep over m_edges: {m_values}")

        for m in m_values:
            print(f"\n  m = {m}: generating network and scanning windows...")

            net_temp = DirectedHomophilicNetwork(
                net.n0,
                net.n_nodes,
                int(m),
                net.h,
                net.f_a,
                net.mu['a'],
                net.mu['b'],
            )
            net_temp.generate_network()

            beta_theory = net_temp.get_beta_for_type(node_type)

            # Build bs and run MC for all windows once
            scan_res = self.scan_over_b(
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

            # Precompute degrees for this m
            degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
            total_edges = degrees.sum() if degrees.size > 0 else 0

            # For each p_c, select largest window from stored windows
            pcs_results = self.select_largest_window_for_pcs(
                windows=windows,
                a=a,
                p_c_list=p_c_list,
            )

            for p_c in p_c_list:
                res_pc = pcs_results[p_c]
                lw = res_pc['largest_window']

                if lw is None:
                    frac_nodes_kept[p_c].append(0.0)
                    frac_edges_kept[p_c].append(0.0)
                    no_window[p_c].append(True)
                    print(f"    No window with p >= p_c={p_c} found.")
                else:
                    b_star = lw['b']
                    # compute fractions
                    if degrees.size == 0 or total_edges == 0:
                        fn = 0.0
                        fe = 0.0
                    else:
                        mask_keep = (degrees >= a) & (degrees <= b_star)
                        fn = mask_keep.mean()
                        fe = degrees[mask_keep].sum() / total_edges

                    frac_nodes_kept[p_c].append(fn)
                    frac_edges_kept[p_c].append(fe)
                    no_window[p_c].append(False)

                    print(
                        f"    p_c={p_c}: largest acceptable window [a={a}, b={b_star}] "
                        f"nodes kept = {fn:.3f}, edges kept = {fe:.3f}"
                    )

        # Prepare figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # horizontal reference line at fraction of new nodes
        frac_new_nodes = net.n_nodes / (net.n_nodes + net.n0)

        colors = plt.cm.viridis(np.linspace(0, 1, len(p_c_list)))

        # Nodes plot
        for color, p_c in zip(colors, p_c_list):
            fn_arr = np.array(frac_nodes_kept[p_c], dtype=float)
            nw_arr = np.array(no_window[p_c], dtype=bool)
            ok_mask = ~nw_arr
            fail_mask = nw_arr

            ax1.plot(
                m_values[ok_mask],
                fn_arr[ok_mask],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=5,
                color=color,
                label=f'p_c = {p_c}',
            )
            ax1.scatter(
                m_values[fail_mask],
                fn_arr[fail_mask],
                marker='x',
                s=50,
                color=color,
            )

        ax1.axhline(
            frac_new_nodes,
            linestyle='--',
            color='black',
            linewidth=1.5,
            label=f'new nodes fraction = {frac_new_nodes:.2f}',
        )
        ax1.set_xlabel('m_edges', fontsize=12)
        ax1.set_ylabel('Fraction of nodes kept', fontsize=12)
        ax1.set_title(
            f'Fraction of nodes with {a} ≤ k ≤ b*(m) (Type "{node_type}")',
            fontsize=13,
            fontweight='bold',
        )
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Edges plot
        for color, p_c in zip(colors, p_c_list):
            fe_arr = np.array(frac_edges_kept[p_c], dtype=float)
            nw_arr = np.array(no_window[p_c], dtype=bool)
            ok_mask = ~nw_arr
            fail_mask = nw_arr

            ax2.plot(
                m_values[ok_mask],
                fe_arr[ok_mask],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=5,
                color=color,
                label=f'p_c = {p_c}',
            )
            ax2.scatter(
                m_values[fail_mask],
                fe_arr[fail_mask],
                marker='x',
                s=50,
                color=color,
            )

        ax2.axhline(
            frac_new_nodes,
            linestyle='--',
            color='black',
            linewidth=1.5,
            label=f'new nodes fraction = {frac_new_nodes:.2f}',
        )
        ax2.set_xlabel('m_edges', fontsize=12)
        ax2.set_ylabel('Fraction of edges (in-degree) kept', fontsize=12)
        ax2.set_title(
            f'Fraction of in-edges on nodes with {a} ≤ k ≤ b*(m) (Type "{node_type}")',
            fontsize=13,
            fontweight='bold',
        )
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        config = {
            'a': a,
            'p_c_list': p_c_list,
            'N_sims': N_sims,
            'b_grid_type': b_grid_type,
            'n_b': n_b,
            'b_min': b_min,
            'b_max': b_max,
        }

        return {
            'm_values': m_values,
            'frac_nodes_kept': frac_nodes_kept,
            'frac_edges_kept': frac_edges_kept,
            'no_window': no_window,
            'windows_info': windows_info,
            'fig': fig,
            'config': config,
        }

class NetworkPlotting:
    """
    Plotting utilities for DirectedHomophilicNetwork.
    Methods take a `DirectedHomophilicNetwork` instance `net`.
    """

    def logarithmic_binning(
        self,
        degrees: np.ndarray,
        bin_factor: float = 1.01
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def plot_degree_distributions(
        self,
        net: DirectedHomophilicNetwork,
        figsize: Tuple = (15, 6),
        discretisations: int = 10**5,
    ):
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
            ax.set_xlabel(r'In-degree $${k^{\mathrm{(in)}}}$$', fontsize=13)
            ax.set_ylabel(r'Probability $${p(k^{\mathrm{(in)}})}$$', fontsize=13)
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

    def plot_in_edge_asymptotes(
        self,
        net: DirectedHomophilicNetwork,
        figsize: Tuple = (10, 6),
    ):
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
            asymptote = net.g_a if type_name == 'a' else net.g_b_asymptotic
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

    def plot_degree_distributions_discrete(
        self,
        net: DirectedHomophilicNetwork,
        figsize: Tuple = (15, 6),
        max_k_display: int = None,
    ):
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
            ax.set_xlabel(r'In-degree $${k^{\mathrm{(in)}}}$$', fontsize=13)
            ax.set_ylabel(r'Probability $${p(k^{\mathrm{(in)}})}$$', fontsize=13)
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
        in_degrees_a, in_degrees_b = net._get_degrees('a'), net._get_degrees('b')

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
            print(
                f"\nType '{type_name}': {len(degrees):,} nodes "
                f"({len(degrees)/n_total*100:.1f}%)"
            )
            print(f"  Mean in-degree: {np.mean(degrees):.2f}")
            print(f"  Max in-degree: {max(degrees) if degrees else 0}")
            print(f"  Min in-degree: {min(degrees) if degrees else 'N/A'}")

        print(
            f"\n_g_a_ = {net.g_a:.6f}, _g_b_ = {net.g_b:.6f}, "
            f"g_b_empirical = {net.g_b_empirical:.6f}"
        )
        print(
            f"g_a + g_b = {net.g_a + net.g_b:.6f} "
            f"(m = {net.m_edges})"
        )

if __name__ == "__main__":
    net = DirectedHomophilicNetwork(n0=50, n_nodes= 1000, m_edges=5, h=0.2, f_a=0.2, mu_a=1, mu_b=5, seed= None,)

    # Helper class instances
    gof = GoFDiagnostics()
    plotting = NetworkPlotting()
    stats = NetworkStatistics()

    start = time.time()
    net.generate_network()
    print(f"Network generated in {time.time() - start:.2f}s")

    Statistics = True
    if Statistics:
        stats.print_statistics(net)

    #net visualization
    Log_Binned = False
    if Log_Binned:
        plotting.plot_degree_distributions(net)
        plt.show()

    Discrete_Linear = False
    if Discrete_Linear:
        plotting.plot_degree_distributions_discrete(net)
        plt.show()

    #Theory Diagnostics
    plot_asymptotes = False
    if plot_asymptotes:
        plotting.plot_in_edge_asymptotes(net)
        plt.show()

    plot_A_const = False
    if plot_A_const:
        plotting.plot_A_values(net)
        plt.show()

    sweep_m_edges_csn = True
    if sweep_m_edges_csn:
        node_type = 'b'
        a = 0
        p_c_list = [0.3, 0.4, 0.5]   # <--- choose whatever set you want
        N_sims = 20
        b_grid_type = 'linear'
        n_b = 5
        b_min = None
        b_max = None

        results_csn_m = gof.csn_sweep_m_edges(net, m_min=2, m_max=25, m_step=2, node_type=node_type, a=a,
            candidate_bs=None, b_min=b_min, b_max=b_max, n_b=n_b, b_grid_type=b_grid_type, N_sims=N_sims,p_c_list=p_c_list,)
        plt.show()