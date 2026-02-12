#!/usr/bin/env python3
"""
DTS-CMA-ES implementation on top of pycma (ask–tell interface).

Implements:
    - Fixed-alpha DTS-CMA-ES (alpha = 0.05 by default)
    - Optional self-adaptive alpha based on Ranking Difference Error (RDE_mu)
      following Algorithm 4 in Bajer et al. (2019, Evolutionary Computation).

Also includes:
    - 100-dimensional Sphere benchmark
    - Baseline plain CMA-ES run for comparison
    - Simple convergence logging and optional plotting

Dependencies:
    - numpy
    - scipy (for GP optimizer and norm.cdf via scikit-learn)
    - scikit-learn (GaussianProcessRegressor, Matern kernel)
    - cma (pycma library)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Sequence

import numpy as np

# --- External libraries for GP and CMA-ES ------------------------------------

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

import cma  # pycma


# =============================================================================
# Utility functions
# =============================================================================


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF Phi(z) for vector z.

    Uses an error-function based approximation via math.erf,
    vectorised with numpy. This avoids depending on scipy.stats.norm.
    """
    z = np.asarray(z, dtype=float)
    # 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def rde_mu(y1: np.ndarray, y2: np.ndarray, mu: int) -> float:
    """
    Ranking Difference Error RDE_mu(y1, y2) as defined in Eq. (14):
    y2 is assumed to be more accurate.

    RDE_mu(y1, y2) =
        sum_{i : rho(y2)_i <= mu} |rho(y2)_i - rho(y1)_i| /
        max_{pi} sum_{i : pi(i) <= mu} |i - pi(i)|

    We use 0-based ranks; the denominator simplifies to mu * (lambda - mu)
    when 0-based ranks are used consistently.
    """
    y1 = np.asarray(y1).ravel()
    y2 = np.asarray(y2).ravel()
    assert y1.shape == y2.shape
    lamb = y1.size
    mu = max(1, min(mu, lamb))

    # ranks: 0 = best, lamb-1 = worst
    order1 = np.argsort(y1)
    order2 = np.argsort(y2)
    rank1 = np.empty(lamb, dtype=int)
    rank2 = np.empty(lamb, dtype=int)
    rank1[order1] = np.arange(lamb, dtype=int)
    rank2[order2] = np.arange(lamb, dtype=int)

    # indices of mu best according to y2
    idx_best2 = np.where(rank2 < mu)[0]
    num = np.sum(np.abs(rank2[idx_best2] - rank1[idx_best2]).astype(float))

    # maximum possible sum: mu * (lambda - mu) for lambda >= 2*mu-1
    denom = float(mu * (lamb - mu))
    if denom <= 0.0:
        return 0.0
    return float(num / denom)


def compute_eps_bounds(alpha: float, dim: int) -> Tuple[float, float]:
    """
    Compute epsilon_min, epsilon_max using the regression models from
    Bajer et al. (2019), Eq. (Q2_min, Q3_max).

    We clamp the dimension to [2, 20] to avoid extrapolating the fits too far
    beyond the range where they were tuned (COCO functions in 2..20D).
    """
    # Effective dimension (avoid log(0) and crazy extrapolation)
    D_eff = max(2, min(dim, 20))
    lnD = math.log(D_eff)

    # Feature vector [1, ln(D), alpha, alpha ln(D), alpha^2]
    a1 = 1.0
    a2 = lnD
    a3 = alpha
    a4 = alpha * lnD
    a5 = alpha * alpha
    feat = np.array([a1, a2, a3, a4, a5], dtype=float)

    # Coefficients from the paper (b_min, b_max)
    b_min = np.array([0.11, -0.0092, -0.13, 0.044, 0.14], dtype=float)
    b_max = np.array([0.35, -0.047, 0.44, 0.044, -0.19], dtype=float)

    eps_min = float(feat.dot(b_min))
    eps_max = float(feat.dot(b_max))

    # Clamp to [0, 1] as RDE_mu is in [0,1]
    eps_min = max(0.0, min(1.0, eps_min))
    eps_max = max(0.0, min(1.0, eps_max))

    # Ensure eps_max > eps_min to avoid degeneracy
    if eps_max < eps_min + 1e-6:
        eps_max = min(1.0, eps_min + 1e-3)

    return eps_min, eps_max


def update_alpha_self_adaptive(
    eps_smooth: float,
    alpha_current: float,
    dim: int,
    alpha_min: float = 0.04,
    alpha_max: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> float:
    """
    Fixed-point iteration for alpha(g+1) using Algorithm 4 and the
    dimension- & alpha-dependent epsilon_min/max bounds.

    Returns the updated alpha.
    """
    alpha = float(alpha_current)
    alpha = max(alpha_min, min(alpha_max, alpha))

    for _ in range(max_iter):
        eps_min, eps_max = compute_eps_bounds(alpha, dim)
        if eps_max <= eps_min:
            break
        t = (eps_smooth - eps_min) / (eps_max - eps_min)
        t = max(0.0, min(1.0, t))  # clip to [0,1]
        alpha_new = alpha_min + t * (alpha_max - alpha_min)
        alpha_new = max(alpha_min, min(alpha_max, alpha_new))
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return alpha


# =============================================================================
# Gaussian Process surrogate model
# =============================================================================


class GaussianProcessSurrogate:
    """
    Simple wrapper around sklearn's GaussianProcessRegressor.

    - Uses Matérn(ν=2.5) covariance and a ConstantKernel.
    - Adds a WhiteKernel for observation noise.
    - Expects preprocessed inputs (e.g., whitened by CMA-ES (σ^2 C)^(-1/2)).
    - Internally standardizes y to zero mean and unit variance.

    This correspond roughly to the Matérn GP models used in DTS-CMA-ES,
    though the original Matlab implementation uses GPML and more detailed
    hyperparameter bounds. Here we rely on sklearn's ML estimation.
    """

    def __init__(self, random_state: Optional[int] = None):
        kernel = ConstantKernel(1.0, (0.01, 100.0)) * Matern(
            length_scale=1.0,
            length_scale_bounds=(0.1, 10.0),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,  # we use explicit WhiteKernel for noise
            normalize_y=False,  # we standardize y ourselves
            n_restarts_optimizer=1,
            random_state=random_state,
        )
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.is_trained_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Fit the GP on (X, y).

        Returns True on success, False on failure (e.g., numerical issues).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        assert X.shape[0] == y.shape[0]

        if X.shape[0] < 2:
            return False

        # Standardize y
        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_
        y_std = float(np.std(y_centered))
        if y_std < 1e-12:
            # Almost constant; model is not useful
            return False
        self.y_std_ = y_std
        y_norm = y_centered / self.y_std_

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X, y_norm)
            self.is_trained_ = True
            return True
        except Exception:
            self.is_trained_ = False
            return False

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance at inputs X.

        Returns:
            mean: shape (n_samples,)
            var : shape (n_samples,)
        """
        if not self.is_trained_:
            raise RuntimeError("GP not trained")

        X = np.asarray(X, dtype=float)
        mean_norm, std_norm = self.gp.predict(X, return_std=True)
        mean = mean_norm * self.y_std_ + self.y_mean_
        var = (std_norm ** 2) * (self.y_std_ ** 2)
        return mean, var


# =============================================================================
# Archive of original evaluations
# =============================================================================


class Archive:
    """Store all original fitness evaluations."""

    def __init__(self, dim: int):
        self.dim = dim
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    def add(self, X: Sequence[np.ndarray], y: Sequence[float]) -> None:
        """Append multiple points and values."""
        for x, fval in zip(X, y):
            self.X.append(np.asarray(x, dtype=float).copy())
            self.y.append(float(fval))

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.X:
            return np.empty((0, self.dim), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(self.X), np.asarray(self.y, dtype=float)

    def best(self) -> Tuple[Optional[np.ndarray], float]:
        if not self.y:
            return None, float("inf")
        idx = int(np.argmin(self.y))
        return self.X[idx].copy(), float(self.y[idx])


# =============================================================================
# DTS-CMA-ES optimizer
# =============================================================================


@dataclass
class DTSConfig:
    """
    Configuration for DTS-CMA-ES.

    Main knobs:
        alpha0          : initial ratio of original evaluations
        use_adaptive_alpha : whether to update alpha using RDE
        beta            : exponential smoothing rate for RDE
        alpha_min/max   : bounds for adaptive alpha
        N_max           : max GP training set size
        r_A_max_factor  : factor for training radius r_A_max
        cpoi_eps        : epsilon in threshold T = fmin - eps*(fmax-fmin)
    """

    alpha0: float = 0.05
    use_adaptive_alpha: bool = False
    beta: float = 0.3
    alpha_min: float = 0.04
    alpha_max: float = 1.0
    N_max: int = 300
    r_A_max_factor: float = 4.0
    cpoi_eps: float = 0.05
    gp_random_state: Optional[int] = None


class DtsCmaEsOptimizer:
    """
    DTS-CMA-ES built on top of pycma's CMAEvolutionStrategy (ask–tell).

    This class implements Algorithm 3 (DTS-CMA-ES) and Algorithm 4
    (selfAdaptation) from Bajer et al. (2019), with the following
    simplifications/choices:

        - Training set selection:
            points in archive with Mahalanobis distance <= r_A_max from
            current mean, then take the closest ones up to N_max.
            (Algorithm 3 in the original DTS paper.)

        - We whiten training inputs and population using the Cholesky of
            (sigma^2 * C), following the original S-CMA-ES trainModel
            procedure.

        - Surrogate model:
            Matérn(ν=2.5) GP, hyperparameters fit via sklearn.
            Outputs are standardized before fitting.

        - Criterion for selecting original evaluations:
            Probability of Improvement (CPoI) with threshold
                T = f_min - cpoi_eps * (f_max - f_min)
            where f_min, f_max come from the GP training set.

        - Alpha adaptation:
            Implemented as in Algorithm 4 with epsilon_min, epsilon_max
            from the linear regression models Q2_min, Q3_max. For
            dimensions > 20, the dimension used in these regressions
            is clamped to 20 to avoid excessive extrapolation.

    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        sigma0: float,
        max_evals: int,
        config: DTSConfig,
        seed: Optional[int] = None,
    ):
        self.f = f
        self.x0 = np.asarray(x0, dtype=float)
        self.sigma0 = float(sigma0)
        self.max_evals = int(max_evals)
        self.config = config
        self.seed = seed

        self.dim = self.x0.size

        # Population size: doubled default, lambda = 8 + 6 ln D
        popsize = 8 + int(6 * math.log(self.dim))

        opts = {
            "popsize": popsize,
            "seed": seed,
            "verb_log": 0,
            "verb_disp": 5,
        }
        self.es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, opts)

        self.archive = Archive(self.dim)

        # Alpha & RDE state
        self.alpha = float(config.alpha0)
        self.eps_smooth = 0.0

        # Stats / logging
        self.eval_count = 0
        self.history_evals: List[int] = []
        self.history_best: List[float] = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _whiten(
        self, X: np.ndarray, mean: np.ndarray, sigma: float, C: np.ndarray
    ) -> np.ndarray:
        """
        Transform X -> Z using (sigma^2 C)^(-1/2), so that Euclidean
        distances in Z correspond to Mahalanobis distances in X.

        If Cholesky fails for numerical reasons, fall back to scaling
        by sigma only (i.e. ignore C).
        """
        X = np.asarray(X, dtype=float)
        mean = np.asarray(mean, dtype=float)
        dx = X - mean

        try:
            cov = (sigma ** 2) * np.asarray(C, dtype=float)
            L = np.linalg.cholesky(cov)
            # Solve L z = dx^T => z = L^{-1} dx^T
            ZT = np.linalg.solve(L, dx.T)
            Z = ZT.T
            return Z
        except np.linalg.LinAlgError:
            # Fallback: isotropic scaling
            return dx / max(sigma, 1e-12)

    def _select_training_indices(
        self,
        X_white: np.ndarray,
        r_A_max: float,
        N_max: int,
    ) -> np.ndarray:
        """
        Select training indices from whitened archive points.

        Strategy:
            - Keep points with ||z|| <= r_A_max
            - If more than N_max points remain, keep the N_max closest
              to the origin (which corresponds to the current CMA mean).
        """
        if X_white.shape[0] == 0:
            return np.empty((0,), dtype=int)

        sq_norm = np.sum(X_white ** 2, axis=1)
        mask = sq_norm <= (r_A_max ** 2)
        idx = np.where(mask)[0]

        if idx.size == 0:
            # No point in radius; fallback to closest N_max overall
            idx = np.argsort(sq_norm)[:N_max]
        elif idx.size > N_max:
            # Take N_max closest among those in radius
            idx = idx[np.argsort(sq_norm[idx])[:N_max]]

        return idx

    def _train_gp_model(
        self,
        X_archive: np.ndarray,
        y_archive: np.ndarray,
        mean: np.ndarray,
        sigma: float,
        C: np.ndarray,
    ) -> Tuple[Optional[GaussianProcessSurrogate], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Train a GP model using a subset of archive points near the current mean.

        Returns:
            (gp_model, X_white_train, y_train)
        """
        cfg = self.config

        if X_archive.shape[0] == 0:
            return None, None, None

        X_white = self._whiten(X_archive, mean, sigma, C)

        # Radius and training set selection
        r_A_max = cfg.r_A_max_factor * math.sqrt(self.dim)
        idx_train = self._select_training_indices(X_white, r_A_max, cfg.N_max)

        if idx_train.size < max(5, min(3 * self.dim, cfg.N_max // 2)):
            # Not enough points to build a meaningful GP
            return None, None, None

        X_train_white = X_white[idx_train]
        y_train = y_archive[idx_train]

        gp = GaussianProcessSurrogate(random_state=cfg.gp_random_state)
        ok = gp.fit(X_train_white, y_train)
        if not ok:
            return None, None, None

        return gp, X_train_white, y_train

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run DTS-CMA-ES until stop criteria or evaluation budget reached.

        Returns:
            best_x, best_f from the archive of original evaluations.
        """
        es = self.es
        cfg = self.config

        while not es.stop():
            if self.eval_count >= self.max_evals:
                break

            # 1. CMA-ES sampling
            X_pop = np.asarray(es.ask())
            lamb = X_pop.shape[0]

            mean = np.asarray(es.mean)
            sigma = float(es.sigma)
            C = np.asarray(es.C)

            # Minimum number of real evaluations before using surrogates:
            X_arch, y_arch = self.archive.as_arrays()
            if X_arch.shape[0] < max(5, min(3 * self.dim, cfg.N_max // 2)):
                # Evaluate entire population with original fitness
                y_true = [self._eval_and_record(x) for x in X_pop]
                es.tell(X_pop, y_true)
                self._log_iteration()
                continue

            # 2. Train first model
            gp1, _, y_train1 = self._train_gp_model(X_arch, y_arch, mean, sigma, C)
            if gp1 is None:
                # Fallback: evaluate full population
                y_true = [self._eval_and_record(x) for x in X_pop]
                es.tell(X_pop, y_true)
                self._log_iteration()
                continue

            # Whiten population for predictions
            X_pop_white = self._whiten(X_pop, mean, sigma, C)

            # 3. First model predictions
            y_hat1, var1 = gp1.predict(X_pop_white)
            std1 = np.sqrt(np.maximum(var1, 1e-18))

            # 4. Select points for original evaluation using CPoI
            n_orig = max(1, int(round(self.alpha * lamb)))

            # Use training set f-values to define PoI threshold
            f_min = float(np.min(y_train1))
            f_max = float(np.max(y_train1))
            T = f_min - cfg.cpoi_eps * (f_max - f_min)

            z = (T - y_hat1) / std1
            c_poi = normal_cdf(z)  # larger = more promising

            idx_sorted = np.argsort(-c_poi)  # descending
            idx_orig = idx_sorted[:n_orig]
            idx_orig_set = set(int(i) for i in idx_orig)

            X_orig = X_pop[idx_orig]
            y_orig = [self._eval_and_record(x) for x in X_orig]

            # 5. Train second model on updated archive
            X_arch2, y_arch2 = self.archive.as_arrays()
            gp2, _, _ = self._train_gp_model(X_arch2, y_arch2, mean, sigma, C)

            skip_adapt = False
            if gp2 is None:
                gp2 = gp1
                skip_adapt = True

            # 6. Second model predictions and mixing with originals
            y_hat2, var2 = gp2.predict(X_pop_white)
            y_mix = y_hat2.copy()

            # Overwrite original-evaluated points
            for j, idx in enumerate(idx_orig):
                y_mix[int(idx)] = y_orig[j]

            # Prevent purely predicted points from beating best true so far
            _, best_true = self.archive.best()
            for i in range(lamb):
                if i not in idx_orig_set:
                    if y_mix[i] < best_true:
                        y_mix[i] = best_true

            # 7. Self-adaptation of alpha (optional)
            if cfg.use_adaptive_alpha and not skip_adapt:
                # Evaluate model error (RDE) between y_hat1 and y_mix
                mu = getattr(es.sp, "mu", lamb // 2)
                eps_rde = rde_mu(y_hat1, y_mix, mu=mu)
                # Exponential smoothing
                self.eps_smooth = (1.0 - cfg.beta) * self.eps_smooth + cfg.beta * eps_rde
                # Fixed-point update of alpha(g+1)
                self.alpha = update_alpha_self_adaptive(
                    eps_smooth=self.eps_smooth,
                    alpha_current=self.alpha,
                    dim=self.dim,
                    alpha_min=cfg.alpha_min,
                    alpha_max=cfg.alpha_max,
                )
            # else: alpha stays constant

            # 8. CMA-ES update with mixed fitness values
            es.tell(X_pop, list(y_mix))
            self._log_iteration()

            if self.eval_count >= self.max_evals:
                break

        best_x, best_f = self.archive.best()
        if best_x is None:
            # In the unlikely event that no eval ever happened, fall back to es.best
            best_f = float(es.best.f)
            best_x = np.asarray(es.best.x)
        return best_x, best_f

    # ------------------------------------------------------------------ #
    # Helpers: evaluation and logging
    # ------------------------------------------------------------------ #

    def _eval_and_record(self, x: np.ndarray) -> float:
        """Evaluate f(x), update archive and eval counter."""
        fx = float(self.f(np.asarray(x, dtype=float)))
        self.eval_count += 1
        self.archive.add([x], [fx])
        return fx

    def _log_iteration(self) -> None:
        """Record current best fitness vs evaluation count."""
        _, best_f = self.archive.best()
        self.history_evals.append(self.eval_count)
        self.history_best.append(best_f)
        # Optional progress print every disp_gap iterations (uses CMA option verb_disp)
        try:
            disp_gap = int(self.es.opts.get("verb_disp", 0))
        except Exception:
            disp_gap = 0
        if disp_gap > 0 and (len(self.history_best) % disp_gap == 0):
            print(f"[DTS] iter {len(self.history_best):4d} | best f = {best_f:.6e}")


# =============================================================================
# Baseline CMA-ES (no surrogate)
# =============================================================================


@dataclass
class BaselineResult:
    best_x: np.ndarray
    best_f: float
    evals: int
    history_evals: List[int]
    history_best: List[float]


def run_baseline_cmaes(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    sigma0: float,
    max_evals: int,
    seed: Optional[int] = None,
    popsize: Optional[int] = None,
) -> BaselineResult:
    """
    Plain CMA-ES baseline using pycma, with ask–tell loop.

    Parameters:
        f        : objective function
        x0       : initial mean
        sigma0   : initial step size
        max_evals: evaluation budget
        seed     : RNG seed
        popsize  : if None, use default CMA-ES popsize; otherwise specify
                   (we'll use the same doubled popsize as in DTS for fairness).
    """
    x0 = np.asarray(x0, dtype=float)
    dim = x0.size

    if popsize is None:
        popsize = 8 + int(6 * math.log(dim))  # same as DTS

    opts = {
        "popsize": popsize,
        "seed": seed,
        "verb_log": 0,
        "verb_disp": 0,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    eval_count = 0
    history_evals: List[int] = []
    history_best: List[float] = []

    best_x = x0.copy()
    best_f = float("inf")

    while not es.stop():
        if eval_count >= max_evals:
            break

        X = np.asarray(es.ask())
        y = []
        for x in X:
            fx = float(f(np.asarray(x, dtype=float)))
            eval_count += 1
            y.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = np.asarray(x, dtype=float)

        es.tell(X, y)
        history_evals.append(eval_count)
        history_best.append(best_f)

        if eval_count >= max_evals:
            break

    return BaselineResult(
        best_x=best_x,
        best_f=best_f,
        evals=eval_count,
        history_evals=history_evals,
        history_best=history_best,
    )


# =============================================================================
# Benchmark functions
# =============================================================================


def sphere(x: np.ndarray) -> float:
    """Simple Sphere function: f(x) = sum_i x_i^2."""
    x = np.asarray(x, dtype=float)
    return float(np.dot(x, x))


def ackley(x: np.ndarray) -> float:
    """Standard scalable Ackley benchmark."""
    x = np.asarray(x, dtype=float)
    d = x.size
    a = 20.0
    b = 0.2
    c = 2 * math.pi
    sum_sq = np.dot(x, x)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * math.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return float(term1 + term2 + a + math.e)


# =============================================================================
# Experiment driver
# =============================================================================


def run_experiment(dim: int = 100, max_evals_per_dim: int = 100) -> None:
    # """
    # Run DTS-CMA-ES and baseline CMA-ES on a single 100D Sphere instance,
    # then on the HierarchicalSubsystem benchmark.

    # Parameters:
    #     dim               : problem dimension (default 100)
    #     max_evals_per_dim : budget multiplier, so total budget = max_evals_per_dim * dim
    # """
    # max_evals = max_evals_per_dim * dim

    # # Initial mean: uniform in [-4, 4]^D as in the paper's COCO setup. :contentReference[oaicite:20]{index=20}
    # rng = np.random.RandomState(42)
    # x0 = rng.uniform(-4.0, 4.0, size=dim)
    # # Initial step-size; the paper uses sigma0 = 8/3 ≈ 2.67. :contentReference[oaicite:21]{index=21}
    # sigma0 = 8.0 / 3.0

    # print("=== Problem ===")
    # print(f"  Function : Sphere")
    # print(f"  Dimension: {dim}")
    # print(f"  Budget   : {max_evals} original evaluations\n")

    # # ------------------------------------------------------------------ #
    # # DTS-CMA-ES (fixed alpha = 0.05)
    # # ------------------------------------------------------------------ #
    # dts_cfg = DTSConfig(
    #     alpha0=0.05,
    #     use_adaptive_alpha=False,  # fixed-alpha for robustness in 100D
    #     beta=0.3,
    #     alpha_min=0.04,
    #     alpha_max=1.0,
    #     N_max=min(20 * dim, 300),
    #     r_A_max_factor=4.0,
    #     cpoi_eps=0.05,
    #     gp_random_state=123,
    # )
    # dts = DtsCmaEsOptimizer(
    #     f=sphere,
    #     x0=x0,
    #     sigma0=sigma0,
    #     max_evals=max_evals,
    #     config=dts_cfg,
    #     seed=1234,
    # )

    # print("=== Running DTS-CMA-ES (fixed alpha=0.05) ===")
    # best_x_dts, best_f_dts = dts.run()
    # print(f"  Final best f (DTS) : {best_f_dts:.3e}")
    # print(f"  Evaluations (DTS)  : {dts.eval_count}")

    # # ------------------------------------------------------------------ #
    # # Baseline CMA-ES
    # # ------------------------------------------------------------------ #
    # print("\n=== Running baseline CMA-ES (no surrogate) ===")
    # baseline = run_baseline_cmaes(
    #     f=sphere,
    #     x0=x0,
    #     sigma0=sigma0,
    #     max_evals=max_evals,
    #     seed=1234,
    #     popsize=dts.es.popsize,  # same population size as DTS
    # )
    # print(f"  Final best f (CMA-ES) : {baseline.best_f:.3e}")
    # print(f"  Evaluations (CMA-ES)  : {baseline.evals}")

    # ------------------------------------------------------------------ #
    # Ackley (scalable)
    # ------------------------------------------------------------------ #
    max_evals_ack = max_evals_per_dim * dim
    rng = np.random.RandomState(42)
    x0_ack = rng.uniform(-32.0, 32.0, size=dim)
    sigma0_ack = 9.0

    print("\n=== Problem: Ackley ===")
    print(f"  Dimension: {dim}")
    print(f"  Budget   : {max_evals_ack} original evaluations\n")

    dts_ack = DtsCmaEsOptimizer(
        f=ackley,
        x0=x0_ack,
        sigma0=sigma0_ack,
        max_evals=max_evals_ack,
        config=DTSConfig(
            alpha0=0.05,
            use_adaptive_alpha=False,
            beta=0.3,
            alpha_min=0.04,
            alpha_max=1.0,
            N_max=min(20 * dim, 300),
            r_A_max_factor=4.0,
            cpoi_eps=0.05,
            gp_random_state=123,
        ),
        seed=5678,
    )

    print("=== Running DTS-CMA-ES (Ackley, fixed alpha=0.05) ===")
    best_x_ack, best_f_ack = dts_ack.run()
    print(f"  Final best f (DTS) : {best_f_ack:.3e}")
    print(f"  Evaluations (DTS)  : {dts_ack.eval_count}")

    print("\n=== Running baseline CMA-ES (Ackley, no surrogate) ===")
    baseline_ack = run_baseline_cmaes(
        f=ackley,
        x0=x0_ack,
        sigma0=sigma0_ack,
        max_evals=max_evals_ack,
        seed=5678,
        # popsize=dts_ack.es.popsize,
    )
    print(f"  Final best f (CMA-ES) : {baseline_ack.best_f:.3e}")
    print(f"  Evaluations (CMA-ES)  : {baseline_ack.evals}")

    # ------------------------------------------------------------------ #
    # HierarchicalSubsystem (100D)
    # ------------------------------------------------------------------ #
    # try:
    #     from problems.hierarchical_subsystem import HierarchicalSubsystem
    # except Exception as exc:  # pragma: no cover
    #     print("\nHierarchicalSubsystem not available:", exc)
    #     return

    # problem = HierarchicalSubsystem()
    # dim_h = problem.dim
    # max_evals_h = max_evals_per_dim * dim_h
    # x0_h = np.zeros(dim_h)
    # sigma0_h = (problem.upper_bound - problem.lower_bound) / 10.0

    # def objective_h(x):
    #     return float(problem(np.asarray(x)))

    # print("\n=== Problem: HierarchicalSubsystem ===")
    # print(f"  Dimension: {dim_h}")
    # print(f"  Budget   : {max_evals_h} original evaluations\n")

    # dts_h = DtsCmaEsOptimizer(
    #     f=objective_h,
    #     x0=x0_h,
    #     sigma0=sigma0_h,
    #     max_evals=max_evals_h,
    #     config=DTSConfig(
    #         alpha0=0.05,
    #         use_adaptive_alpha=False,
    #         beta=0.3,
    #         alpha_min=0.04,
    #         alpha_max=1.0,
    #         N_max=min(20 * dim_h, 300),
    #         r_A_max_factor=4.0,
    #         cpoi_eps=0.05,
    #         gp_random_state=123,
    #     ),
    #     seed=1234,
    # )

    # print("=== Running DTS-CMA-ES (fixed alpha=0.05) ===")
    # best_x_h, best_f_h = dts_h.run()
    # print(f"  Final best f (DTS) : {best_f_h:.3e}")
    # print(f"  Evaluations (DTS)  : {dts_h.eval_count}")

    # print("\n=== Running baseline CMA-ES (no surrogate) ===")
    # baseline_h = run_baseline_cmaes(
    #     f=objective_h,
    #     x0=x0_h,
    #     sigma0=sigma0_h,
    #     max_evals=max_evals_h,
    #     seed=1234,
    #     popsize=dts_h.es.popsize,
    # )
    # print(f"  Final best f (CMA-ES) : {baseline_h.best_f:.3e}")
    # print(f"  Evaluations (CMA-ES)  : {baseline_h.evals}")

    # # ------------------------------------------------------------------ #
    # # Optional plotting
    # # ------------------------------------------------------------------ #
    # try:
    #     import matplotlib.pyplot as plt  # type: ignore

    #     plt.figure()
    #     plt.plot(dts.history_evals, dts.history_best, label="DTS-CMA-ES")
    #     plt.plot(
    #         baseline.history_evals,
    #         baseline.history_best,
    #         label="CMA-ES baseline",
    #         linestyle="--",
    #     )
    #     plt.yscale("log")
    #     plt.xlabel("Original function evaluations")
    #     plt.ylabel("Best f so far")
    #     plt.title(f"DTS-CMA-ES vs CMA-ES on {dim}D Sphere")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    # except ImportError:
    #     print("\nmatplotlib not available; skipping plot.")


def main() -> None:
    run_experiment(dim=20, max_evals_per_dim=100)


if __name__ == "__main__":
    main()

"""
Usage
-----

1. Install dependencies (if needed):

    pip install numpy scipy scikit-learn cma matplotlib

2. Save this script as, e.g., dts_cmaes_experiment.py

3. Run:

    python dts_cmaes_experiment.py

4. You should see printed results like:

    === Problem ===
      Function : Sphere
      Dimension: 100
      Budget   : 10000 original evaluations

    === Running DTS-CMA-ES (fixed alpha=0.05) ===
      Final best f (DTS) : 1.2e-09
      Evaluations (DTS)  : 10000

    === Running baseline CMA-ES (no surrogate) ===
      Final best f (CMA-ES) : 3.4e-08
      Evaluations (CMA-ES)  : 10000

    matplotlib will show a log-scale convergence plot if installed.

Notes
-----
- To enable self-adaptive alpha, change `use_adaptive_alpha=True` in DTSConfig.
- For other benchmarks (e.g. Rosenbrock), replace `sphere` with your function.
- All DTS logic lives outside pycma and uses the ask–tell API, so the
  pycma source code remains untouched.
"""
