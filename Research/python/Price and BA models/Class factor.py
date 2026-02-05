import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func, beta as beta_func
from scipy.stats import chisquare, ksone
from typing import Dict, Tuple, List
import time
from scipy.stats import chi2
from scipy.optimize import minimize
from scipy.special import beta as beta_func

class DirectedHomophilicNetwork:
    """Optimized directed network with homophilic preferential attachment."""

    # Theory and Model.

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


    def csn_distance(
        self,
        net: DirectedHomophilicNetwork,
        data,
        a: int,
        b: int,
        beta,
        node_type: str,
    ) -> float:
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

        data = np.asarray(data, dtype=int)
        if data.size == 0:
            # Convention: no data in [a,b] => distance 0
            return 0.0

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
        # STEP 4: pass beta into the CDF call
        S_n = self.theoretical_cdf_discrete(
            net,
            n_vals,
            node_type,
            kmin=a,
            kmax=b,
            beta=beta,      # <-- this is the important addition
        )
        S_n = np.asarray(S_n, dtype=float)

        # Denominator sqrt(S (1-S)), with safeguard
        denom = np.sqrt(S_n * (1.0 - S_n))
        valid = denom > 0

        if not np.any(valid):
            # If everything is degenerate (S in {0,1}), define D=0
            return 0.0

        numer = np.abs(empirical_ratio[valid] - S_n[valid])
        D_vals = numer / denom[valid]
        D = float(np.max(D_vals))
        return D

    def _fit_beta_mle(
        self,
        net: DirectedHomophilicNetwork,
        data,
        a: int,
        b: int,
        node_type: str,
        beta_init,
    ):
        """
        Curve-fit (p0, alpha, gamma) for the given node_type on the window [a,b]
        by matching the truncated PMF to the empirical PMF (least-squares in
        log-probabilities).

        Parameters
        ----------
        net : DirectedHomophilicNetwork
            Network whose asymptotic parameters provide a natural initial guess.
        data : array-like
            Degrees, already truncated to a <= x <= b for the chosen node_type.
        a, b : int
            Window [a,b] for truncation.
        node_type : str
            'a' or 'b'.
        beta_init :
            Initial guess for (p0, alpha, gamma). Can be:
              - a tuple/list/array of length 3, or
              - None, in which case we use net._get_params(node_type).

        Returns
        -------
        beta_mle : np.ndarray, shape (3,)
            Fitted (p0, alpha, gamma).
        """

        data = np.asarray(data, dtype=int)
        if data.size == 0:
            # No data in window: fall back to initial guess from net
            params0 = net._get_params(node_type)
            return np.array(
                [params0['p0'], params0['alpha'], params0['gamma']], dtype=float
            )

        # Get initial guess for (p0, alpha, gamma)
        params0 = net._get_params(node_type)
        if beta_init is None:
            p0_0 = float(params0['p0'])
            alpha0 = float(params0['alpha'])
            gamma0 = float(params0['gamma'])
        else:
            beta_init_arr = np.asarray(beta_init, dtype=float)
            if beta_init_arr.shape[0] != 3:
                raise ValueError(
                    "beta_init for _fit_beta_mle should be length-3: (p0, alpha, gamma)"
                )
            p0_0, alpha0, gamma0 = beta_init_arr

        # Support on [a,b]
        k_support = np.arange(a, b + 1, dtype=int)

        # Empirical PMF on [a,b]
        unique_k, counts = np.unique(data, return_counts=True)
        total = counts.sum()
        emp_pmf = counts / total

        def truncated_pmf(p0: float, alpha: float, gamma: float) -> np.ndarray:
            """
            Build truncated PMF p(k | p0, alpha, gamma) over k in [a,b],
            with renormalisation on [a,b].
            """
            # Enforce basic constraints
            if not (0.0 < p0 < 1.0) or alpha <= 0 or gamma <= 0:
                # Return tiny uniform; will be penalised
                return np.full_like(k_support, 1e-300, dtype=float)

            pmf = np.zeros_like(k_support, dtype=float)

            # k == 0 if in window
            zero_mask = (k_support == 0)
            if np.any(zero_mask):
                pmf[zero_mask] = p0

            # k > 0 part
            pos_mask = (k_support > 0)
            if np.any(pos_mask):
                k_pos = k_support[pos_mask]
                # Form: p(k) ∝ p0 * B(k, alpha + gamma) / B(k, alpha)
                pmf[pos_mask] = (
                    p0
                    * beta_func(k_pos, alpha + gamma)
                    / beta_func(k_pos, alpha)
                )

            # Renormalise over [a,b]
            Z_trunc = pmf.sum()
            if Z_trunc <= 0:
                return np.full_like(k_support, 1e-300, dtype=float)

            pmf /= Z_trunc
            pmf = np.clip(pmf, 1e-300, 1.0)
            return pmf

        def objective(theta: np.ndarray) -> float:
            """
            Least-squares in log-PMF between empirical and model on [a,b].
            We only compare at k where we have empirical mass.
            """
            p0, alpha, gamma = theta
            model_pmf_full = truncated_pmf(p0, alpha, gamma)

            model_probs = []
            emp_probs = []
            for k, p_emp in zip(unique_k, emp_pmf):
                if a <= k <= b:
                    idx = np.where(k_support == k)[0]
                    if idx.size == 0:
                        continue
                    model_probs.append(model_pmf_full[idx[0]])
                    emp_probs.append(p_emp)

            if not model_probs:
                # No overlap between support and data; large penalty
                return 1e6

            model_probs = np.asarray(model_probs, dtype=float)
            emp_probs = np.asarray(emp_probs, dtype=float)

            # Compare in log-space
            log_model = np.log(model_probs)
            log_emp = np.log(emp_probs)
            return np.sum((log_model - log_emp) ** 2)

        # Initial theta
        theta0 = np.array([p0_0, alpha0, gamma0], dtype=float)

        # Bounds: 0 < p0 < 1, alpha > 0, gamma > 0
        bounds = [
            (1e-6, 1.0 - 1e-6),  # p0
            (1e-4, None),        # alpha
            (1e-4, None),        # gamma
        ]

        res = minimize(
            objective,
            theta0,
            method='L-BFGS-B',
            bounds=bounds,
        )

        if not res.success:
            # If optimisation fails, fall back to initial guess
            p0_mle, alpha_mle, gamma_mle = p0_0, alpha0, gamma0
        else:
            p0_mle, alpha_mle, gamma_mle = res.x

        return np.array([p0_mle, alpha_mle, gamma_mle], dtype=float)

    def mc_on_window(
        self,
        net: DirectedHomophilicNetwork,
        node_type: str,
        a: int,
        b: int,
        beta_theory,
        N_sims: int,
    ):
        """
        Monte Carlo experiment on a single window [a,b] for a given node_type,
        with fixed theoretical parameters beta_theory.

        For each simulation s = 1,...,N_sims:

            1) Generate a network from the model with the same parameters as 'net'
               (which encode beta_theory at the global/asymptotic level).
            2) Extract degrees for node_type and truncate to [a,b].
            3) Compute D_theory_s = D(data_s; beta_theory; [a,b]) using csn_distance.
               (Currently 'beta_theory' is not passed into the CDF; the model
                structure is taken from net_sim, but we keep beta_theory in the
                interface.)
            4) Fit beta_mle_s = (p0, alpha, gamma) on the truncated data via
               _fit_beta_mle, and compute
               D_mle_s = D(data_s; beta_mle_s; [a,b]).

        After N_sims, compute:

            p([a,b])      = (# sims with D_mle_s > D_theory_s) / N_sims
            sigma_p([a,b]) = sqrt( p (1 - p) / N_sims )

        Returns
        -------
        result : dict
            {
              'a': a,
              'b': b,
              'D_theory': np.ndarray,   # shape (N_sims,)
              'D_mle': np.ndarray,      # shape (N_sims,)
              'beta_mle': np.ndarray,   # shape (N_sims, 3) if numeric
              'p': float,
              'sigma_p': float,
              'beta_mle_mean': np.ndarray or None,
              'beta_mle_std': np.ndarray or None,
            }
        """

        D_theory_list = []
        D_mle_list = []
        beta_mle_list = []

        # Ensure beta_theory has the shape we expect for comparison/storage.
        # For now we just store it; csn_distance uses net_sim's internal params.
        beta_theory_arr = np.asarray(beta_theory) if beta_theory is not None else None

        for s in range(N_sims):
            # 1) Generate a new network using the same parameters as 'net'
            net_sim = DirectedHomophilicNetwork(
                net.n0,
                net.n_nodes,
                net.m_edges,
                net.h,
                net.f_a,
                net.mu['a'],
                net.mu['b'],
                seed=None,  # random seed can be left free or controlled externally
            )
            net_sim.generate_network()

            # 2) Extract degrees and truncate to [a,b]
            degrees = np.array(net_sim._get_degrees(node_type), dtype=int)
            data_s = degrees[(degrees >= a) & (degrees <= b)]

            if data_s.size == 0:
                # No data in [a,b]: define distances as 0 and beta_mle = beta_theory
                D_theory_list.append(0.0)
                D_mle_list.append(0.0)
                if beta_theory_arr is not None:
                    beta_mle_list.append(beta_theory_arr)
                else:
                    beta_mle_list.append(np.array([np.nan, np.nan, np.nan]))
                continue

            # 3) Compute D_theory_s with the CSN distance
            #    Note: csn_distance currently uses net_sim's theoretical CDF,
            #    which encodes the global asymptotic parameters. 'beta_theory'
            #    is kept in the interface for future use when you plug beta
            #    directly into the model.
            D_theory_s = self.csn_distance(
                net_sim,
                data_s,
                a,
                b,
                beta_theory,
                node_type,
            )

            # 4) Fit beta_mle_s = (p0, alpha, gamma) on [a,b] and compute D_mle_s
            beta_mle_s = self._fit_beta_mle(
                net_sim,
                data_s,
                a,
                b,
                node_type,
                beta_init=None,  # start from net_sim._get_params(node_type)
            )

            # For D_mle, we still use csn_distance; the current implementation
            # of csn_distance obtains the theoretical CDF from net_sim itself,
            # so beta_mle_s is stored for inspection but not yet passed into the
            # CDF machinery. Once you refactor net to accept (p0, alpha, gamma)
            # explicitly, you can wire beta_mle_s into csn_distance.
            D_mle_s = self.csn_distance(
                net_sim,
                data_s,
                a,
                b,
                beta_mle_s,
                node_type,
            )

            D_theory_list.append(D_theory_s)
            D_mle_list.append(D_mle_s)
            beta_mle_list.append(beta_mle_s)

        # Convert to arrays
        D_theory_arr = np.array(D_theory_list, dtype=float)
        D_mle_arr = np.array(D_mle_list, dtype=float)

        # p([a,b]) = fraction with D_mle > D_theory
        greater_mask = D_mle_arr > D_theory_arr
        p = float(np.mean(greater_mask.astype(float)))
        sigma_p = float(np.sqrt(p * (1.0 - p) / max(N_sims, 1)))

        # Try to build numeric array for beta_mle and compute mean/std
        try:
            beta_mle_arr = np.asarray(beta_mle_list, dtype=float)
            beta_mle_mean = np.nanmean(beta_mle_arr, axis=0)
            beta_mle_std = np.nanstd(beta_mle_arr, axis=0)
        except Exception:
            beta_mle_arr = np.array(beta_mle_list, dtype=object)
            beta_mle_mean = None
            beta_mle_std = None

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
        }
        return result

    def scan_over_b(
        self,
        net: DirectedHomophilicNetwork,
        node_type: str,
        a: int,
        beta_theory,
        candidate_bs,
        N_sims: int,
    ):
        """
        Scan over multiple upper cutoffs b_j for a fixed lower cutoff 'a'
        and node_type, running the CSN/AD-style Monte Carlo per window.

        Parameters
        ----------
        net : DirectedHomophilicNetwork
            Base network whose parameters define the theoretical model.
        node_type : str
            'a' or 'b'.
        a : int
            Lower cutoff of the window [a, b_j] (fixed across all windows).
        beta_theory :
            Theoretical parameter vector for the window model. Currently
            stored and returned but not yet wired into the CDF computation,
            since the model is still parameterised by 'net'.
        candidate_bs : array-like of int
            Sequence of upper cutoffs b_j, with a < b_j <= max degree.
        N_sims : int
            Number of Monte Carlo simulations to perform per window.

        Returns
        -------
        result : dict
            {
              'a': a,
              'beta_theory': beta_theory,
              'windows': [
                  {
                    'b': b_j,
                    'D_theory': np.ndarray,   # shape (N_sims,)
                    'D_mle': np.ndarray,      # shape (N_sims,)
                    'beta_mle': np.ndarray,   # shape (N_sims, 3) if numeric
                    'p': float,
                    'sigma_p': float,
                    'beta_mle_mean': np.ndarray or None,
                    'beta_mle_std': np.ndarray or None,
                  },...
              ]
            }
        """
        candidate_bs = list(candidate_bs)
        windows_results = []

        for b in candidate_bs:
            if b <= a:
                raise ValueError(
                    f"Each b_j must satisfy b_j > a. Got a={a}, b_j={b}."
                )

            window_result = self.mc_on_window(
                net=net,
                node_type=node_type,
                a=a,
                b=int(b),
                beta_theory=beta_theory,
                N_sims=N_sims,
            )

            # Attach b explicitly into the window dict for convenience
            window_result_with_b = dict(window_result)
            window_result_with_b['b'] = int(b)

            windows_results.append(window_result_with_b)

        result = {
            'a': a,
            'beta_theory': beta_theory,
            'windows': windows_results,
        }
        return result

    def scan_until_threshold(
        self,
        net: DirectedHomophilicNetwork,
        node_type: str,
        a: int,
        beta_theory,
        candidate_bs,
        N_sims: int,
        p_c: float,
    ):
        """
        Scan over upper cutoffs b_j for fixed a and node_type,
        starting from the largest window and shrinking until we find
        the smallest truncation that still satisfies p([a,b_j]) >= p_c.

        New behaviour
        -------------
        - Let {b_1,..., b_J} be candidate_bs sorted ascending.
        - We evaluate windows in order: [a, b_J], [a, b_{J-1}],..., [a, b_1].
        - We stop at the first b_j where p([a,b_j]) >= p_c.
        - That b_j is the smallest truncation of the range [a, max b_j]
          that still passes the threshold, i.e. the largest range we can
          keep while satisfying p >= p_c after removing as much tail as needed.

        Parameters
        ----------
        net : DirectedHomophilicNetwork
            Base network whose parameters define the model.
        node_type : str
            'a' or 'b'.
        a : int
            Lower cutoff of the window [a, b_j].
        beta_theory :
            Theoretical parameter vector (p0, alpha, gamma) for this node_type.
        candidate_bs : array-like of int
            Sequence of upper cutoffs b_j (not necessarily sorted).
        N_sims : int
            Number of Monte Carlo simulations per window.
        p_c : float
            Threshold in [0,1]. We look for the smallest b_j (when scanning
            from max to min) such that p([a,b_j]) >= p_c.

        Returns
        -------
        result : dict
            {
              'a': a,
              'beta_theory': beta_theory,
              'p_c': p_c,
              'windows_evaluated': [  # in the order evaluated (from largest to smallest b)
                  {
                    'b': b_j,
                    'D_theory':...,
                    'D_mle':...,
                    'beta_mle':...,
                    'p': p([a,b_j]),
                    'sigma_p':...,
                    'beta_mle_mean':...,
                    'beta_mle_std':...,
                  },...
              ],
              'largest_window': {
                  'a': a,
                  'b': b_star,
                  'p': p_star,
                  'sigma_p': sigma_p_star,
                  'index': j_star,    # index into windows_evaluated
              } or None if no window satisfies p >= p_c
            }
        """
        # Sort candidate b_j ascending, then we will traverse from largest to smallest
        bs_sorted = sorted(int(b) for b in candidate_bs)
        windows_evaluated = []
        largest_window_info = None  # will hold the first window (from top) with p >= p_c

        # Traverse from largest b downward
        for j_rev, b in enumerate(reversed(bs_sorted)):
            if b <= a:
                raise ValueError(
                    f"Each b_j must satisfy b_j > a. Got a={a}, b_j={b}."
                )

            # Run MC on this window [a,b]
            window_result = self.mc_on_window(
                net=net,
                node_type=node_type,
                a=a,
                b=int(b),
                beta_theory=beta_theory,
                N_sims=N_sims,
            )

            # Attach 'b' explicitly
            window_result['b'] = int(b)
            windows_evaluated.append(window_result)

            p_val = window_result['p']

            # We scan from largest to smallest; the first window with p >= p_c
            # is the *largest range that passes* after truncating as much as needed.
            if p_val >= p_c:
                largest_window_info = {
                    'a': a,
                    'b': int(b),
                    'p': float(p_val),
                    'sigma_p': float(window_result['sigma_p']),
                    'index': j_rev,
                }
                break  # stop shrinking once we find a window that passes

        result = {
            'a': a,
            'beta_theory': beta_theory,
            'p_c': float(p_c),
            'windows_evaluated': windows_evaluated,  # note: order is from largest to smaller b's actually tested
            'largest_window': largest_window_info,
        }
        return result

    def csn_sweep_m_edges(
        self,
        net: DirectedHomophilicNetwork,
        m_min: int,
        m_max: int,
        m_step: int = 1,
        node_type: str = 'b',
        a: int = 0,
        # --- b_j grid control ---
        candidate_bs=None,
        b_min: int = None,
        b_max: int = None,
        n_b: int = 20,
        b_grid_type: str = 'linear',  # 'linear' or 'log'
        # --- Monte Carlo control ---
        N_sims: int = 20,
        p_c: float = 0.1,
        figsize=(12, 6),
    ):
        """
        CSN-based sweep over m_edges.

        For each m in [m_min, m_max] with step m_step:
            1) Generate a network with that m.
            2) Build a grid of upper cutoffs b_j according to:
                 - candidate_bs (if provided), or
                 - [b_min, b_max, n_b, b_grid_type].
            3) Run scan_until_threshold on windows [a, b_j], with N_sims MC sims
               per window, and threshold p_c.
            4) Record the largest window [a, b*(m)] with p([a,b*(m)]) >= p_c.

        Plots:
            - Left: window length L*(m) = b*(m) - a + 1 vs m_edges,
            - Right: p([a,b*(m)]) vs m_edges.

        Parameters
        ----------
        net : DirectedHomophilicNetwork
            Provides base parameters n0, n_nodes, h, f_a, mu_a, mu_b.
        m_min, m_max : int
            Range of m_edges values to explore.
        m_step : int
            Step in m_edges.
        node_type : str
            'a' or 'b'.
        a : int
            Lower cutoff of the window [a, b_j] (fixed for all m).
        candidate_bs : array-like of int or None
            If not None, use this exact sequence of b_j for all m.
        b_min, b_max : int or None
            If candidate_bs is None, these define the range for b_j.
            Defaults:
                b_min = a + 1
                b_max = max degree of node_type in that network.
        n_b : int
            Number of b_j points to generate when candidate_bs is None.
        b_grid_type : {'linear', 'log'}
            How to space the b_j values between b_min and b_max.
        N_sims : int
            Number of Monte Carlo simulations per truncation window.
        p_c : float
            Threshold: we enlarge [a,b] until p([a,b]) < p_c.
        figsize : tuple
            Figure size for the diagnostic plot.

        Returns
        -------
        result : dict
            {
              'm_values': np.ndarray,
              'b_star': list of b*(m) or np.nan,
              'L_star': list of L*(m) = b*(m) - a + 1 or np.nan,
              'p_star': list of p([a,b*(m)]) or np.nan,
              'sigma_p_star': list of sigma_p([a,b*(m)]) or np.nan,
              'windows_info': list of scan_until_threshold results per m,
              'fig': matplotlib.figure.Figure,
              'config': {
                  'a': a,
                  'p_c': p_c,
                  'N_sims': N_sims,
                  'b_grid_type': b_grid_type,
                  'n_b': n_b,...
              }
            }
        """
        m_values = np.arange(m_min, m_max + 1, m_step, dtype=int)
        b_star_list = []
        L_star_list = []
        p_star_list = []
        sigma_p_star_list = []
        windows_info = []

        print(f"\nCSN sweep over m_edges: {m_values}")

        for m in m_values:
            print(f"\n  m = {m}: generating network and scanning windows...")

            # 1) Generate a network with this m
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

            # Theory beta from this network's asymptotics for the node_type
            beta_theory = net_temp.get_beta_for_type(node_type)

            # 2) Build the b_j grid
            if candidate_bs is not None:
                bs = np.array(candidate_bs, dtype=int)
            else:
                degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                if degrees.size == 0:
                    # No nodes of this type; record NaNs and continue
                    b_star_list.append(np.nan)
                    L_star_list.append(np.nan)
                    p_star_list.append(np.nan)
                    sigma_p_star_list.append(np.nan)
                    windows_info.append(None)
                    print("    No nodes of this type; skipping.")
                    continue

                max_deg = int(degrees.max())

                # Determine b_min and b_max if not provided
                b_min_eff = a + 1 if b_min is None else max(a + 1, b_min)
                b_max_eff = max_deg if b_max is None else min(max_deg, b_max)

                if b_min_eff > b_max_eff:
                    # No valid b range; skip
                    b_star_list.append(np.nan)
                    L_star_list.append(np.nan)
                    p_star_list.append(np.nan)
                    sigma_p_star_list.append(np.nan)
                    windows_info.append(None)
                    print("    No valid b range (b_min_eff > b_max_eff); skipping.")
                    continue

                if b_grid_type == 'linear':
                    bs = np.linspace(b_min_eff, b_max_eff, n_b, dtype=int)
                    bs = np.unique(bs)  # ensure sorted + unique
                elif b_grid_type == 'log':
                    # Log grid in [b_min_eff, b_max_eff]
                    bs = np.logspace(
                        np.log10(b_min_eff),
                        np.log10(b_max_eff),
                        n_b,
                        dtype=int,
                    )
                    bs = np.unique(bs)
                else:
                    raise ValueError(
                        f"Unknown b_grid_type='{b_grid_type}'. Use 'linear' or 'log'."
                    )

            # 3) Run scan_until_threshold on [a, b_j]
            scan_res = self.scan_until_threshold(
                net=net_temp,
                node_type=node_type,
                a=a,
                beta_theory=beta_theory,
                candidate_bs=bs,
                N_sims=N_sims,
                p_c=p_c,
            )
            windows_info.append(scan_res)

            lw = scan_res['largest_window']
            if lw is None:
                # No acceptable window (all p < p_c)
                b_star_list.append(np.nan)
                L_star_list.append(np.nan)
                p_star_list.append(np.nan)
                sigma_p_star_list.append(np.nan)
                print("    No window with p >= p_c found.")
            else:
                b_star = lw['b']
                L_star = b_star - a + 1  # inclusive window length
                b_star_list.append(b_star)
                L_star_list.append(L_star)
                p_star_list.append(lw['p'])
                sigma_p_star_list.append(lw['sigma_p'])
                print(
                    f"    Largest acceptable window: [a={lw['a']}, b={b_star}] "
                    f"(L={L_star}), p = {lw['p']:.3f} ± {lw['sigma_p']:.3f}"
                )

        # Convert to arrays for plotting
        b_star_arr = np.array(b_star_list, dtype=float)
        L_star_arr = np.array(L_star_list, dtype=float)
        p_star_arr = np.array(p_star_list, dtype=float)
        sigma_p_star_arr = np.array(sigma_p_star_list, dtype=float)

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: window length L*(m) vs m
        ax1.plot(
            m_values,
            L_star_arr,
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=6,
        )
        ax1.set_xlabel('m_edges', fontsize=12)
        ax1.set_ylabel('Window length L* = b* - a + 1', fontsize=12)
        ax1.set_title(
            f'Largest Window Length vs m_edges (Type "{node_type}")',
            fontsize=13,
            fontweight='bold',
        )
        ax1.grid(True, alpha=0.3)

        # Right: corresponding p([a,b*(m)]) vs m
        ax2.errorbar(
            m_values,
            p_star_arr,
            yerr=sigma_p_star_arr,
            fmt='o-',
            linewidth=2,
            markersize=6,
            capsize=4,
            label=r'$p([a,b^*])$',
        )
        ax2.axhline(p_c, linestyle='--', color='black', alpha=0.5, label=f'p_c = {p_c}')
        ax2.set_xlabel('m_edges', fontsize=12)
        ax2.set_ylabel(r'$p([a,b^*])$', fontsize=12)
        ax2.set_title(
            f'p-value at Largest Window vs m_edges (Type "{node_type}")',
            fontsize=13,
            fontweight='bold',
        )
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        config = {
            'a': a,
            'p_c': p_c,
            'N_sims': N_sims,
            'b_grid_type': b_grid_type,
            'n_b': n_b,
            'b_min': b_min,
            'b_max': b_max,
        }

        return {
            'm_values': m_values,
            'b_star': b_star_list,
            'L_star': L_star_list,
            'p_star': p_star_list,
            'sigma_p_star': sigma_p_star_list,
            'windows_info': windows_info,
            'fig': fig,
            'config': config,
        }


    def ks_test_and_plot(self, net: DirectedHomophilicNetwork, node_type='b', kmin=0, kmax=25, figsize=(12, 6)):
        """
        KS test with visualization over integer degrees in [kmin, kmax].
        Only the KS test decides the range; theoretical CDF is evaluated at these points.
        """

        degrees = np.array(net._get_degrees(node_type), dtype=int)
        if len(degrees) == 0:
            print("No nodes of this type!")
            return None, None, None

        if kmax is None:
            kmax = int(degrees.max())

        # Filter degrees to the test range
        degrees_filtered = degrees[(degrees >= kmin) & (degrees <= kmax)]
        if len(degrees_filtered) == 0:
            print(f"No degrees in range [{kmin}, {kmax}]")
            return None, None, None

        n = len(degrees_filtered)
        percent_used = 100 * n / len(degrees)

        # Compute empirical CDF over filtered integers (vectorized)
        unique_degrees = np.arange(kmin, kmax + 1)
        sorted_data = np.sort(degrees_filtered)
        counts = np.searchsorted(sorted_data, unique_degrees, side='right')
        empirical_cdf = counts / n

        # Theoretical CDF values at unique degrees
        theoretical_cdf_vals = self.theoretical_cdf_discrete(
            net, unique_degrees, node_type, kmin=kmin, kmax=kmax
        )

        # KS statistic
        discrepancies = np.abs(empirical_cdf - theoretical_cdf_vals)
        D = discrepancies.max()
        p_value = ksone.sf(D, n)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            unique_degrees,
            empirical_cdf,
            'o-',
            label='Empirical CDF',
            color='blue',
            alpha=0.7,
        )
        ax.plot(
            unique_degrees,
            theoretical_cdf_vals,
            '-',
            label='Theoretical CDF',
            color='red',
            alpha=0.7,
        )

        ax.set_xlabel('In-degree k')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(
            f'KS Test: Type "{node_type}" (D={D:.4f}, p={p_value:.2e})'
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=kmin, right=kmax)
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()

        print(
            f"\nKS Test: Type '{node_type}', k∈[{kmin},{kmax}], {percent_used:.1f}% data"
        )
        print(f"  D = {D:.6f}, p = {p_value:.6e}")

        return fig, D, p_value

    def aggregate_pvals_diagnostic(self, pvals):
        pvals = np.asarray(pvals)
        R = len(pvals)
        if R == 0:
            return np.nan, np.nan, np.nan
        p_median = np.median(pvals)
        fisher_stat = -2.0 * np.sum(np.log(pvals))
        p_fisher = 1.0 - chi2.cdf(fisher_stat, df=2 * R)
        p_hmp = R / np.sum(1.0 / pvals)
        return p_median, p_fisher, p_hmp

    def ks_sweep_m_edges(
        self,
        net: DirectedHomophilicNetwork,
        m_min: int,
        m_max: int,
        m_step: int = 1,
        node_type='b',
        kmin: int = 0,
        kmax: int = 25,
        n_runs: int = 3,
        figsize=(12, 6),
    ):
        """
        Sweep over m_edges, compute KS statistics and diagnostic p-values.
        Returns dict with D medians, median p, Fisher p, HMP p, and figure.
        """

        m_values = np.arange(m_min, m_max + 1, m_step)
        D_medians, p_medians, p_fishers, p_hmps = [], [], [], []

        print(f"\nSweeping m_edges: {m_values}")

        for m in m_values:
            D_runs, p_runs = [], []

            for _ in range(n_runs):
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

                degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                if degrees.size == 0:
                    continue

                kmax_actual = int(degrees.max()) if kmax is None else kmax
                degrees = degrees[(degrees >= kmin) & (degrees <= kmax_actual)]
                if degrees.size == 0:
                    continue

                n = len(degrees)
                support = np.arange(kmin, kmax_actual + 1)
                sorted_deg = np.sort(degrees)
                empirical_cdf = (
                    np.searchsorted(sorted_deg, support, side='right') / n
                )

                theoretical_cdf = self.theoretical_cdf_discrete(
                    net_temp, support, node_type, kmin=kmin, kmax=kmax_actual
                )

                D = np.max(np.abs(empirical_cdf - theoretical_cdf))
                p = ksone.sf(D, n)

                D_runs.append(D)
                p_runs.append(p)

            # Compute diagnostics
            if D_runs:
                D_median = np.median(D_runs)
                p_median, p_fisher, p_hmp = self.aggregate_pvals_diagnostic(p_runs)

                D_medians.append(D_median)
                p_medians.append(p_median)
                p_fishers.append(p_fisher)
                p_hmps.append(p_hmp)

                print(
                    f"  m={m:2d}: D_med={D_median:.4f}, "
                    f"p_med={p_median:.3f}, "
                    f"p_F={p_fisher:.3f}, "
                    f"p_HMP={p_hmp:.3f}"
                )
            else:
                D_medians.append(np.nan)
                p_medians.append(np.nan)
                p_fishers.append(np.nan)
                p_hmps.append(np.nan)

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(m_values[: len(D_medians)], D_medians, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('m_edges', fontsize=12)
        ax1.set_ylabel('KS Statistic D (median)', fontsize=12)
        ax1.set_title(
            f'KS D Statistic Median vs m_edges (Type "{node_type}")',
            fontsize=13,
            fontweight='bold',
        )
        ax1.grid(True, alpha=0.3)

        ax2.plot(m_values[: len(p_medians)], p_medians, marker='o', label='Median p')
        ax2.plot(m_values[: len(p_fishers)], p_fishers, marker='s', label='Fisher p')
        ax2.plot(m_values[: len(p_hmps)], p_hmps, marker='^', label='HMP p')
        ax2.axhline(0.05, linestyle='--', color='black', alpha=0.5)
        ax2.set_xlabel('m_edges', fontsize=12)
        ax2.set_ylabel('p-value', fontsize=12)
        ax2.set_title(
            f'p-value Diagnostics vs m_edges (Type "{node_type}")',
            fontsize=13,
            fontweight='bold',
        )
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        return {
            'm_values': m_values[: len(D_medians)],
            'D_medians': D_medians,
            'p_medians': p_medians,
            'p_fishers': p_fishers,
            'p_hmps': p_hmps,
            'fig': fig,
        }

    def ks_sweep_n0(
        self,
        net: DirectedHomophilicNetwork,
        n0_min,
        n0_max,
        n0_step=1,
        node_type='b',
        kmin=0,
        kmax=25,
        n_runs=3,
        figsize=(12, 6),
    ):
        """
        Sweep over initial node count n0 for diagnostic KS testing.
        Returns median, Fisher, and harmonic mean p-values across runs.
        """
        n0_values = np.arange(n0_min, n0_max + 1, n0_step)
        D_medians, p_medians, p_fishers, p_hmps = [], [], [], []

        print(f"\nSweeping n0: {n0_values}")

        for n0 in n0_values:
            D_runs, p_runs = [], []

            for _ in range(n_runs):
                net_temp = DirectedHomophilicNetwork(
                    int(n0),
                    net.n_nodes,
                    net.m_edges,
                    net.h,
                    net.f_a,
                    net.mu['a'],
                    net.mu['b'],
                )
                net_temp.generate_network()

                degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                if degrees.size == 0:
                    continue

                kmax_actual = int(degrees.max()) if kmax is None else kmax
                degrees = degrees[(degrees >= kmin) & (degrees <= kmax_actual)]
                if degrees.size == 0:
                    continue

                n = len(degrees)
                support = np.arange(kmin, kmax_actual + 1)
                sorted_deg = np.sort(degrees)
                empirical_cdf = (
                    np.searchsorted(sorted_deg, support, side='right') / n
                )

                theoretical_cdf = self.theoretical_cdf_discrete(
                    net_temp, support, node_type, kmin=kmin, kmax=kmax_actual
                )

                D = np.max(np.abs(empirical_cdf - theoretical_cdf))
                p = ksone.sf(D, n)

                D_runs.append(D)
                p_runs.append(p)

            if D_runs:
                D_median = np.median(D_runs)
                p_median, p_fisher, p_hmp = self.aggregate_pvals_diagnostic(p_runs)

                D_medians.append(D_median)
                p_medians.append(p_median)
                p_fishers.append(p_fisher)
                p_hmps.append(p_hmp)

                print(
                    f"  n0={n0:3d}: D_med={D_median:.4f}, "
                    f"p_med={p_median:.3f}, "
                    f"p_F={p_fisher:.3f}, "
                    f"p_HMP={p_hmp:.3f}"
                )
            else:
                D_medians.append(np.nan)
                p_medians.append(np.nan)
                p_fishers.append(np.nan)
                p_hmps.append(np.nan)

        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(n0_values[: len(D_medians)], p_medians, marker='o', label='Median p')
        ax.plot(
            n0_values[: len(p_fishers)],
            p_fishers,
            marker='s',
            label='Fisher p',
        )
        ax.plot(n0_values[: len(p_hmps)], p_hmps, marker='^', label='HMP p')
        ax.axhline(0.05, linestyle='--', color='black', alpha=0.5, label='p=0.05')
        ax.set_xlabel('n0 (initial nodes)', fontsize=12)
        ax.set_ylabel('p-value', fontsize=12)
        ax.set_title(
            f'p-value Diagnostics vs n0 (Type "{node_type}")',
            fontsize=13,
            fontweight='bold',
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        return {
            'n0_values': n0_values[: len(D_medians)],
            'D_medians': D_medians,
            'p_medians': p_medians,
            'p_fishers': p_fishers,
            'p_hmps': p_hmps,
            'fig': fig,
        }

    def ks_surface_n0_m(
        self,
        net: DirectedHomophilicNetwork,
        n0_min,
        n0_max,
        n0_step,
        m_min,
        m_max,
        m_step,
        node_type='b',
        kmin=0,
        kmax=25,
        n_runs=3,
        figsize=(16, 6),
    ):
        """
        dict : Results containing grid values and p-value/D statistic surfaces
        """
        import warnings
        warnings.filterwarnings('ignore')
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        n0_values = np.arange(n0_min, n0_max + 1, n0_step)
        m_values = np.arange(m_min, m_max + 1, m_step)

        # Create meshgrid for surface
        N0_grid, M_grid = np.meshgrid(n0_values, m_values, indexing='ij')
        P_grid = np.zeros_like(N0_grid, dtype=float)
        D_grid = np.zeros_like(N0_grid, dtype=float)

        print(f"\nCreating surface: n0={n0_values}, m={m_values}")
        print(f"Fixed: n_nodes={net.n_nodes}, h={net.h}, f_a={net.f_a}")
        print(f"Total points: {len(n0_values) * len(m_values)}\n")

        point_count = 0
        total_points = len(n0_values) * len(m_values)

        for i, n0 in enumerate(n0_values):
            for j, m in enumerate(m_values):
                D_runs, p_runs = [], []

                for run in range(n_runs):
                    # Generate network
                    net_temp = DirectedHomophilicNetwork(
                        int(n0),
                        net.n_nodes,
                        int(m),
                        net.h,
                        net.f_a,
                        net.mu['a'],
                        net.mu['b'],
                    )
                    net_temp.generate_network()

                    # KS test
                    degrees = np.array(net_temp._get_degrees(node_type), dtype=int)
                    if len(degrees) == 0:
                        continue

                    # Auto-set kmax if None
                    kmax_actual = int(degrees.max()) if kmax is None else kmax

                    degrees_filtered = degrees[
                        (degrees >= kmin) & (degrees <= kmax_actual)
                    ]
                    if len(degrees_filtered) == 0:
                        continue

                    n = len(degrees_filtered)
                    unique_degrees = np.arange(kmin, kmax_actual + 1)
                    sorted_data = np.sort(degrees_filtered)
                    counts = np.searchsorted(sorted_data, unique_degrees, side='right')
                    empirical_cdf = counts / n

                    theoretical_cdf_vals = self.theoretical_cdf_discrete(
                        net_temp,
                        unique_degrees,
                        node_type,
                        kmin=kmin,
                        kmax=kmax_actual,
                    )

                    D = np.abs(empirical_cdf - theoretical_cdf_vals).max()
                    p_value = ksone.sf(D, n)

                    D_runs.append(D)
                    p_runs.append(p_value)

                if len(D_runs) > 0:
                    P_grid[i, j] = np.mean(p_runs)
                    D_grid[i, j] = np.mean(D_runs)
                else:
                    P_grid[i, j] = np.nan
                    D_grid[i, j] = np.nan

                point_count += 1
                if point_count % 5 == 0 or point_count == total_points:
                    print(
                        f"  Progress: {point_count}/{total_points} "
                        f"({100*point_count/total_points:.1f}%)"
                    )

        # Create visualizations
        fig = plt.figure(figsize=figsize)

        # 3D surface for p-value
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(
            N0_grid,
            M_grid,
            P_grid,
            cmap='viridis',
            edgecolor='none',
            alpha=0.8,
        )
        ax1.set_xlabel('n0', fontsize=11)
        ax1.set_ylabel('m_edges', fontsize=11)
        ax1.set_zlabel('p-value', fontsize=11)
        ax1.set_title(
            f'KS p-value Surface (Type "{node_type}")',
            fontsize=12,
            fontweight='bold',
        )
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

        # 3D surface for D statistic
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(
            N0_grid,
            M_grid,
            D_grid,
            cmap='plasma',
            edgecolor='none',
            alpha=0.8,
        )
        ax2.set_xlabel('n0', fontsize=11)
        ax2.set_ylabel('m_edges', fontsize=11)
        ax2.set_zlabel('KS Statistic D', fontsize=11)
        ax2.set_title(
            f'KS D Statistic Surface (Type "{node_type}")',
            fontsize=12,
            fontweight='bold',
        )
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

        plt.tight_layout()

        # Create contour plot as well
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

        # p-value contour
        contour1 = ax3.contourf(M_grid, N0_grid, P_grid, levels=20, cmap='viridis')
        ax3.contour(
            M_grid,
            N0_grid,
            P_grid,
            levels=[0.05],
            colors='red',
            linewidths=2,
            linestyles='--',
        )
        ax3.set_xlabel('m_edges', fontsize=12)
        ax3.set_ylabel('n0', fontsize=12)
        ax3.set_title(
            f'p-value Contours (Type "{node_type}")\nRed line: p=0.05',
            fontsize=12,
            fontweight='bold',
        )
        fig2.colorbar(contour1, ax=ax3)

        # D statistic contour
        contour2 = ax4.contourf(M_grid, N0_grid, D_grid, levels=20, cmap='plasma')
        ax4.set_xlabel('m_edges', fontsize=12)
        ax4.set_ylabel('n0', fontsize=12)
        ax4.set_title(
            f'KS D Statistic Contours (Type "{node_type}")',
            fontsize=12,
            fontweight='bold',
        )
        fig2.colorbar(contour2, ax=ax4)

        plt.tight_layout()

        return {
            'n0_grid': N0_grid,
            'm_grid': M_grid,
            'p_grid': P_grid,
            'd_grid': D_grid,
            'fig_3d': fig,
            'fig_contour': fig2,
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

    def plot_degree_distributions_hybrid(
        self,
        net: DirectedHomophilicNetwork,
        figsize: Tuple = (15, 12),
        max_k_display: int = None,
    ):
        """
        Plot using discrete integer probabilities - TWO VIEWS.
        Top row: log-log for full range
        Bottom row: linear scale zoomed to where you have good statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        for idx, (node_type, color) in enumerate([('a', 'red'), ('b', 'blue')]):
            in_degrees = net._get_degrees(node_type)
            if len(in_degrees) == 0:
                continue

            # Empirical PMF at integer values
            unique_k, counts = np.unique(in_degrees, return_counts=True)
            empirical_pmf = counts / len(in_degrees)

            # Theoretical curve at integer values
            k_max = int(np.max(in_degrees)) if max_k_display is None else max_k_display
            k_range = np.arange(0, k_max + 1)
            theo_probs = net.theoretical_distribution(k_range, node_type)

            # TOP ROW: Log-log (full range)
            ax_log = axes[0, idx]
            ax_log.scatter(
                unique_k,
                empirical_pmf,
                s=50,
                alpha=0.7,
                color=color,
                edgecolors='black',
                linewidths=0.5,
                label='Simulation',
                zorder=3,
            )
            mask = theo_probs > 0
            ax_log.plot(
                k_range[mask],
                theo_probs[mask],
                '-',
                linewidth=2.5,
                color='dark' + color,
                alpha=0.85,
                label='Theory',
                zorder=2,
            )
            ax_log.set_xscale('log')
            ax_log.set_yscale('log')
            ax_log.set_xlabel(r'In-degree $${k^{\mathrm{(in)}}}$$', fontsize=11)
            ax_log.set_ylabel(r'Probability $${p(k^{\mathrm{(in)}})}$$', fontsize=11)
            ax_log.set_title(
                f'Type "{node_type}" - LOG SCALE (full range)',
                fontsize=11,
                fontweight='bold',
            )
            ax_log.legend(fontsize=9)
            ax_log.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)

            # BOTTOM ROW: Linear (zoomed to good statistics)
            ax_lin = axes[1, idx]
            # Only show k where you have at least 5 observations
            good_stats_mask = counts >= 5
            k_cutoff = unique_k[good_stats_mask][-1] if np.any(good_stats_mask) else 50
            k_cutoff = min(k_cutoff, 100)  # Cap at k=100 for readability

            # Plot empirical
            plot_mask = unique_k <= k_cutoff
            ax_lin.scatter(
                unique_k[plot_mask],
                empirical_pmf[plot_mask],
                s=50,
                alpha=0.7,
                color=color,
                edgecolors='black',
                linewidths=0.5,
                label='Simulation',
                zorder=3,
            )

            # Plot theory up to cutoff
            k_range_zoom = np.arange(0, k_cutoff + 1)
            theo_probs_zoom = net.theoretical_distribution(k_range_zoom, node_type)
            ax_lin.plot(
                k_range_zoom,
                theo_probs_zoom,
                '-',
                linewidth=2.5,
                color='dark' + color,
                alpha=0.85,
                label='Theory',
                zorder=2,
            )

            ax_lin.set_xlabel(r'In-degree $${k^{\mathrm{(in)}}}$$', fontsize=11)
            ax_lin.set_ylabel(r'Probability $${p(k^{\mathrm{(in)}})}$$', fontsize=11)
            ax_lin.set_title(
                f'Type "{node_type}" - LINEAR (k ≤ {k_cutoff}, good stats)',
                fontsize=11,
                fontweight='bold',
            )
            ax_lin.legend(fontsize=9)
            ax_lin.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_lin.set_xlim(-1, k_cutoff + 5)

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
    net = DirectedHomophilicNetwork(
        n0=50,
        n_nodes=2500,
        m_edges=5,
        h=0.8,
        f_a=0.8,
        mu_a=5,
        mu_b=1,
        seed= None,
    )

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

    Hybrid_Plot =False
    if Hybrid_Plot:
        plotting.plot_degree_distributions_hybrid(net)
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


    #new ks test.
    sweep_m_edges_csn = True
    if sweep_m_edges_csn:
        node_type = 'b'
        a = 0             # lhs truncation parameter a (must be < b_j)            
        p_c = 0.4         # significance threshold
        N_sims = 50        # MC sims per truncation window
        b_grid_type = 'linear'  # 'linear' or 'log'
        n_b = 50           # number of b_j grid points
        b_min = 1          # minimal b_j (must be > a)
        b_max = None       # None => auto-use max degree for each network

        results_csn_m = gof.csn_sweep_m_edges(
            net,
            m_min=2,
            m_max=30,
            m_step=1,
            node_type=node_type,
            a=a,
            candidate_bs=None,      # use automatically built grid
            b_min=b_min,
            b_max=b_max,
            n_b=n_b,
            b_grid_type=b_grid_type,
            N_sims=N_sims,
            p_c=p_c,
        )
        plt.show()



    KS_test = False
    if KS_test:
        fig, D, p = gof.ks_test_and_plot(
            net, node_type='b', kmin=0, kmax=None
        )
        plt.show()

    sweep_m_edges = False
    if sweep_m_edges:
        results_m = gof.ks_sweep_m_edges(
            net,
            m_min=2,
            m_max=60,
            m_step=2,
            node_type='b',
            kmin=0,
            kmax=None,
            n_runs=10,
        )
        plt.show()

    sweep_n0 = False
    if sweep_n0:
        results_n0 = gof.ks_sweep_n0(
            net,
            n0_min=50,
            n0_max=500,
            n0_step=50,
            node_type='b',
            kmin=0,
            kmax=25,
            n_runs=3,
        )
        plt.show()

    surface_n0_m = False
    if surface_n0_m:
        results_surface = gof.ks_surface_n0_m(
            net,
            n0_min=50,
            n0_max=500,
            n0_step=100,
            m_min=5,
            m_max=20,
            m_step=5,
            node_type='b',
            kmin=0,
            kmax=None,
            n_runs=2,
        )
        plt.show()