from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root on sys.path when run as a script.
if __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

try:
    import cma  # pycma
except Exception as exc:  # pragma: no cover
    cma = None  # type: ignore
    _cma_import_error = exc

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

try:
    # Package import style (recommended)
    from .base import BaseOptimizer
    from .surrogate import DTSSurrogateRanker
except Exception:  # pragma: no cover
    # Script-mode fallback (allows running this file directly)
    from base import BaseOptimizer  # type: ignore
    from surrogate import DTSSurrogateRanker  # type: ignore


# =============================================================================
# Numeric helpers (Py3.7 compatible)
# =============================================================================

def _norm_ppf(p: float) -> float:
    """
    Approximate inverse CDF (PPF) of standard normal distribution.
    Peter J. Acklam's rational approximation. No scipy required.
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")

    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den

    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return (num / den) * q


def _chi2_ppf_wilson_hilferty(p: float, df: int) -> float:
    """
    Approximate chi-square quantile via Wilson–Hilferty transform:
        Chi2_ppf(p, df) ≈ df * (1 - 2/(9df) + z*sqrt(2/(9df)))^3
    """
    df = max(1, int(df))
    p = float(p)
    p = min(max(p, 1e-12), 1.0 - 1e-12)

    z = _norm_ppf(p)
    a = 2.0 / (9.0 * df)
    q = df * (1.0 - a + z * math.sqrt(a)) ** 3
    return float(max(q, 1e-12))


def _rde_mu(y1: np.ndarray, y2: np.ndarray, mu: int) -> float:
    """
    Ranking Difference Error RDE_mu(y1, y2), Bajer et al. (2019), Eq. 14.
    y2 assumed more accurate. Uses 0-based ranks. Denominator = mu*(lambda-mu).
    """
    y1 = np.asarray(y1, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()
    if y1.shape != y2.shape:
        raise ValueError("y1 and y2 must have same shape")

    lamb = int(y1.size)
    if lamb < 2:
        return 0.0

    mu = max(1, min(int(mu), lamb))

    order1 = np.argsort(y1)
    order2 = np.argsort(y2)
    rank1 = np.empty(lamb, dtype=int)
    rank2 = np.empty(lamb, dtype=int)
    rank1[order1] = np.arange(lamb, dtype=int)
    rank2[order2] = np.arange(lamb, dtype=int)

    idx_best2 = np.where(rank2 < mu)[0]
    num = float(np.sum(np.abs(rank2[idx_best2] - rank1[idx_best2])))

    denom = float(mu * (lamb - mu))
    return 0.0 if denom <= 0.0 else float(num / denom)


def _compute_eps_bounds(alpha: float, dim: int, clamp_dim_to_20: bool = True) -> Tuple[float, float]:
    """
    Bajer et al. (2019) regression models Q2_min, Q3_max for epsilon bounds.
    Clamp dim to 20 (paper tuned on 2..20D COCO) by default.
    """
    D_eff = int(dim)
    if clamp_dim_to_20:
        D_eff = max(2, min(D_eff, 20))
    else:
        D_eff = max(2, D_eff)

    lnD = math.log(D_eff)
    feat = np.array([1.0, lnD, alpha, alpha * lnD, alpha * alpha], dtype=float)

    b_min = np.array([0.11, -0.0092, -0.13, 0.044, 0.14], dtype=float)
    b_max = np.array([0.35, -0.047, 0.44, 0.044, -0.19], dtype=float)

    eps_min = float(feat.dot(b_min))
    eps_max = float(feat.dot(b_max))

    eps_min = max(0.0, min(1.0, eps_min))
    eps_max = max(0.0, min(1.0, eps_max))
    if eps_max < eps_min + 1e-6:
        eps_max = min(1.0, eps_min + 1e-3)

    return eps_min, eps_max


def _update_alpha_self_adaptive(
    eps_smooth: float,
    alpha_current: float,
    dim: int,
    alpha_min: float,
    alpha_max: float,
    clamp_dim_to_20: bool = True,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> float:
    """Algorithm 4 fixed-point iteration (eps bounds depend on alpha)."""
    alpha = float(alpha_current)
    alpha = max(alpha_min, min(alpha_max, alpha))

    for _ in range(int(max_iter)):
        eps_min, eps_max = _compute_eps_bounds(alpha, dim, clamp_dim_to_20=clamp_dim_to_20)
        if eps_max <= eps_min:
            break
        t = (eps_smooth - eps_min) / (eps_max - eps_min)
        t = max(0.0, min(1.0, t))
        alpha_new = alpha_min + t * (alpha_max - alpha_min)
        alpha_new = max(alpha_min, min(alpha_max, alpha_new))
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return float(alpha)


# =============================================================================
# Small internal structures (archive, model bundle, GP wrapper)
# =============================================================================

@dataclass
class _ModelBundle:
    gp: "_GaussianProcessSurrogate"
    mean: np.ndarray
    sigma: float
    C: np.ndarray
    y_train_min: float
    y_train_max: float
    age: int = 0


class _Archive:
    """Store all true evaluations collected so far."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    @property
    def size(self) -> int:
        return len(self.y)

    def add_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        for xi, fi in zip(X, y):
            self.X.append(np.asarray(xi, dtype=float).copy())
            self.y.append(float(fi))

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.X:
            return np.empty((0, self.dim), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(self.X), np.asarray(self.y, dtype=float)

    def best(self) -> Tuple[Optional[np.ndarray], float]:
        if not self.y:
            return None, float("inf")
        i = int(np.argmin(self.y))
        return self.X[i].copy(), float(self.y[i])


class _GaussianProcessSurrogate:
    """
    sklearn GP wrapper.
    - Expects whitened inputs (Sigma^{-1/2}(x-mean)).
    - Standardizes y before fitting.
    """

    def __init__(
        self,
        dim: int,
        nu: float,
        constant_value: float,
        constant_bounds: Tuple[float, float],
        length_scale: float,
        length_scale_bounds: Tuple[float, float],
        noise_level: float,
        noise_bounds: Tuple[float, float],
        n_restarts_optimizer: int,
        random_state: Optional[int],
        y_std_min: float,
    ) -> None:
        self.dim = int(dim)
        self.y_std_min = float(y_std_min)

        kernel = (
            ConstantKernel(constant_value, constant_bounds)
            * Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
            + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_bounds)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=False,
            n_restarts_optimizer=int(n_restarts_optimizer),
            random_state=random_state,
        )
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.is_trained_: bool = False

    def fit(self, Xw: np.ndarray, y: np.ndarray) -> bool:
        Xw = np.asarray(Xw, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if Xw.shape[0] != y.shape[0] or Xw.shape[0] < 2:
            return False

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_
        y_std = float(np.std(y_centered))
        if y_std < self.y_std_min:
            return False

        self.y_std_ = y_std
        y_norm = y_centered / self.y_std_

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                self.gp.fit(Xw, y_norm)
            self.is_trained_ = True
            return True
        except Exception:
            self.is_trained_ = False
            return False

    def predict(self, Xw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained_:
            raise RuntimeError("GP not trained")
        Xw = np.asarray(Xw, dtype=float)
        mean_norm, std_norm = self.gp.predict(Xw, return_std=True)
        mean = mean_norm * self.y_std_ + self.y_mean_
        var = (std_norm ** 2) * (self.y_std_ ** 2)
        return mean, var


# =============================================================================
# DTSCMAESOptimizer (library-standard API)
# =============================================================================

class DTSCMAESOptimizer(BaseOptimizer):
    """
    DTS-CMA-ES (Doubly Trained Surrogate CMA-ES) wrapper around pycma (ask/tell).

    Budget counts ONLY true evaluations.
    """

    def __init__(
        self,
        dim: int,                         # Problem dimensionality D (number of decision variables)
        seed: int = 0,                    # Random seed (affects CMA-ES sampling and some model randomness)
        # ---- CMA-ES init ----
        sigma0: float = 3.0,              # Initial CMA-ES global step-size σ0 (sampling scale; larger = more exploration)
        x0: Optional[np.ndarray] = None,  # Initial CMA-ES mean m0 (None -> default to all-zeros vector)
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # Variable bounds (lower, upper) passed to CMA-ES
        pop_size: Optional[int] = None,   # CMA-ES population size λ (None -> derived from popsize_mode formula)
        popsize_mode: str = "default",  # "default" / "double"
        cma_options: Optional[Dict[str, Any]] = None,  # Extra pycma options (e.g., bounds, stopping tolerances, etc.)
        # ---- DTS knobs ----
        alpha0: float = 0.05,             # Initial true-evaluation fraction α: n_orig = ceil(alpha * λ)
        # alpha0: float = 0.1,
        use_adaptive_alpha: bool = False, # Enable self-adaptive α (Bajer et al. 2019) based on ranking error ε
        beta: float = 0.3,                # Smoothing factor for ε: eps_smooth=(1-beta)*eps_smooth + beta*eps_rde
        alpha_min: float = 0.04,          # Lower bound for adaptive α (prevents too few true evaluations)
        alpha_max: float = 1.0,           # Upper bound for adaptive α (1.0 -> full true-evaluation generation)
        min_true_per_gen: int = 1,        # Minimum number of true evaluations per generation (hard floor)
        # warmup_min_points: Optional[int] = None,  # default: max(10D, 50)
        warmup_min_points: Optional[int] = 1000,  # default: max(10D, 50)
        n_min_train: Optional[int] = None,        # default: max(5, 3D)
        # n_max_train: Optional[int] = None,        # default: min(20D, 300)
        n_max_train: Optional[int] = 1000,        # default: min(20D, 300)
        radius_mode: str = "chi2",                # "chi2" or "sqrt_dim"
        radius_chi2_p: float = 0.99,              # Chi-square quantile p used to define the Mahalanobis-radius
        r_A_max_factor: float = 4.0,              # Multiplicative factor to scale the radius (larger -> less local training set)
        selection_criterion: str = "cstd",        # "cstd"/"mean"/"cpoi"/"cei"
        cpoi_eps: float = 0.05,                   # Improvement threshold/epsilon used by PoI/EI-style criteria (ranker-specific)
        prediction_guard: str = "shift",          # "shift"/"clamp"/"none"
        use_model_cache: bool = True,             # Reuse last trained model if (re)training fails (avoid full-eval fallback)
        max_model_age: int = 2,                   # Maximum number of generations a cached model can be reused
        # whiten_jitter: float = 1e-12,             # Diagonal jitter added to covariance in whitening for numerical stability
        whiten_jitter: float = 1e-10,             # Diagonal jitter added to covariance in whitening for numerical stability

        # ---- NEW: local-only surrogate gate ----
        require_min_points_in_radius: bool = False,  # If True, require >= n_min_train archive points inside radius to enable surrogate
        # ---- GP hyperparams ----
        gp_nu: float = 2.5,                          # Matérn kernel ν parameter (smoothness; 2.5 is a common default)
        gp_constant_value: float = 1.0,              # Initial ConstantKernel value (overall output scale prior)
        gp_constant_bounds: Tuple[float, float] = (1e-3, 1e3),  # Bounds for ConstantKernel during hyperparameter optimization
        gp_length_scale: float = 1,                # Initial Matérn length-scale (input scale prior)
        gp_length_scale_bounds: Tuple[float, float] = (1e-3, 1e2),  # Bounds for length-scale during hyperparameter optimization
        gp_noise_level: float = 1e-6,                # Initial WhiteKernel noise level (observation/numerical noise)
        gp_noise_bounds: Tuple[float, float] = (1e-10, 1e-3),       # Bounds for noise level during hyperparameter optimization
        # gp_n_restarts_optimizer: int = 1,            # Number of restarts for GP hyperparameter optimizer (more = slower, potentially better)
        gp_n_restarts_optimizer: int = 10,            # Number of restarts for GP hyperparameter optimizer (more = slower, potentially better)

        gp_random_state: Optional[int] = None,       # Random state for GP hyperparameter optimization (None -> not fixed)
        gp_y_std_min: float = 1e-12,                 # Minimum std(y) required to fit GP (too-flat y -> skip training)
        # ---- Logging ----
        print_every: int = 5,                      # Print progress every N generations
        **kwargs: Any,                               # Extra unused keyword args (kept for interface compatibility)
    ) -> None:

        # Avoid forwarding bounds/cma_options to BaseOptimizer
        super().__init__(dim=dim, seed=seed)

        # CMA / init
        self.sigma0 = float(sigma0)
        if x0 is None:
            self.x0 = np.zeros(dim, dtype=float)   # 默认原点
        else:
            self.x0 = np.asarray(x0, dtype=float).reshape(dim)
        self.bounds = bounds
        self.pop_size = pop_size
        self.popsize_mode = str(popsize_mode).lower()
        self.cma_options = dict(cma_options) if cma_options else {}
        if self.bounds is not None and "bounds" not in self.cma_options:
            self.cma_options["bounds"] = self.bounds

        # DTS params
        self.alpha0 = float(alpha0)
        self.use_adaptive_alpha = bool(use_adaptive_alpha)
        self.beta = float(beta)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.min_true_per_gen = int(min_true_per_gen)

        # Dimension-dependent defaults (match ref script)
        n_min = int(n_min_train or max(5, 3 * dim))
        n_max = int(n_max_train or min(20 * dim, 300))
        warmup = int(warmup_min_points or max(10 * dim, 50))
        if n_max < n_min:
            n_max = n_min
        if warmup < n_min:
            warmup = n_min

        self.warmup_min_points = warmup
        self.n_min_train = n_min
        self.n_max_train = n_max

        self.radius_mode = str(radius_mode).lower()
        self.radius_chi2_p = float(radius_chi2_p)
        self.r_A_max_factor = float(r_A_max_factor)

        self.selection_criterion = str(selection_criterion).lower()
        self.cpoi_eps = float(cpoi_eps)

        self.prediction_guard = str(prediction_guard).lower()
        self.use_model_cache = bool(use_model_cache)
        self.max_model_age = int(max_model_age)
        self.whiten_jitter = float(whiten_jitter)

        # NEW: gate surrogate use by local archive density
        self.require_min_points_in_radius = bool(require_min_points_in_radius)

        # GP params
        self.gp_nu = float(gp_nu)
        self.gp_constant_value = float(gp_constant_value)
        self.gp_constant_bounds = tuple(gp_constant_bounds)
        self.gp_length_scale = float(gp_length_scale)
        self.gp_length_scale_bounds = tuple(gp_length_scale_bounds)
        self.gp_noise_level = float(gp_noise_level)
        self.gp_noise_bounds = tuple(gp_noise_bounds)
        self.gp_n_restarts_optimizer = int(gp_n_restarts_optimizer)
        self.gp_random_state = gp_random_state
        self.gp_y_std_min = float(gp_y_std_min)

        # Ranker (decoupled in surrogate.py)
        self.ranker = DTSSurrogateRanker(mode=self.selection_criterion, cpoi_eps=self.cpoi_eps)

        # Printing
        self.print_every = int(print_every)

        # Outputs after optimize()
        self.stop_reasons_: Dict[str, Any] = {}
        self.evals_: int = 0
        self.history_: List[float] = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_popsize(self) -> int:
        if self.pop_size is not None:
            return int(self.pop_size)
        d = max(2, int(self.dim))
        if self.popsize_mode == "default":
            return 4 + int(math.floor(3.0 * math.log(d)))
        if self.popsize_mode == "double":
            return 8 + int(math.ceil(6.0 * math.log(d)))
            # return 2*(8 + int(math.ceil(6.0 * math.log(d))))
        raise ValueError(f"Unknown popsize_mode: {self.popsize_mode}")

    def _compute_r_A_max(self) -> float:
        if self.radius_mode == "sqrt_dim":
            return float(self.r_A_max_factor * math.sqrt(self.dim))
        if self.radius_mode == "chi2":
            q = _chi2_ppf_wilson_hilferty(self.radius_chi2_p, self.dim)
            return float(self.r_A_max_factor * math.sqrt(q))
        raise ValueError(f"Unknown radius_mode: {self.radius_mode}")

    def _whiten(self, X: np.ndarray, mean: np.ndarray, sigma: float, C: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mean = np.asarray(mean, dtype=float)
        dx = X - mean
        if dx.ndim == 1:
            dx = dx[None, :]

        C = np.asarray(C, dtype=float)
        C = 0.5 * (C + C.T)
        cov = (sigma ** 2) * C
        cov = 0.5 * (cov + cov.T)
        cov = cov + self.whiten_jitter * np.eye(self.dim)

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, self.whiten_jitter, None)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        return np.linalg.solve(L, dx.T).T

    def _select_training_indices(self, X_white: np.ndarray, r_A_max: float) -> np.ndarray:
        n_total = int(X_white.shape[0])
        if n_total == 0:
            return np.empty((0,), dtype=int)

        sq_norm = np.sum(X_white ** 2, axis=1)
        inside = np.where(sq_norm <= (r_A_max ** 2))[0]

        if inside.size < self.n_min_train:
            k = min(self.n_max_train, n_total)
            return np.argsort(sq_norm)[:k].astype(int)

        if inside.size > self.n_max_train:
            inside = inside[np.argsort(sq_norm[inside])[: self.n_max_train]]
        return inside.astype(int)

    def _train_gp_model(
        self,
        X_archive: np.ndarray,
        y_archive: np.ndarray,
        mean: np.ndarray,
        sigma: float,
        C: np.ndarray,
    ) -> Optional[_ModelBundle]:
        if X_archive.shape[0] < self.n_min_train:
            return None

        X_white = self._whiten(X_archive, mean, sigma, C)
        r_A_max = self._compute_r_A_max()
        idx = self._select_training_indices(X_white, r_A_max)
        if idx.size < self.n_min_train:
            return None

        X_train = X_white[idx]
        y_train = y_archive[idx]

        gp = _GaussianProcessSurrogate(
            dim=self.dim,
            nu=self.gp_nu,
            constant_value=self.gp_constant_value,
            constant_bounds=self.gp_constant_bounds,
            length_scale=self.gp_length_scale,
            length_scale_bounds=self.gp_length_scale_bounds,
            noise_level=self.gp_noise_level,
            noise_bounds=self.gp_noise_bounds,
            n_restarts_optimizer=self.gp_n_restarts_optimizer,
            random_state=self.gp_random_state,
            y_std_min=self.gp_y_std_min,
        )
        ok = gp.fit(X_train, y_train)
        if not ok:
            return None

        return _ModelBundle(
            gp=gp,
            mean=np.asarray(mean, dtype=float).copy(),
            sigma=float(sigma),
            C=np.asarray(C, dtype=float).copy(),
            y_train_min=float(np.min(y_train)),
            y_train_max=float(np.max(y_train)),
            age=0,
        )

    @staticmethod
    def _guard_shift(y_mix: np.ndarray, idx_orig_set: set, best_true: float) -> np.ndarray:
        y = np.asarray(y_mix, dtype=float).copy()
        mask_pred = np.array([i not in idx_orig_set for i in range(y.size)], dtype=bool)
        if not np.any(mask_pred):
            return y
        min_pred = float(np.min(y[mask_pred]))
        if min_pred < best_true:
            y[mask_pred] = y[mask_pred] + (best_true - min_pred)
        return y

    @staticmethod
    def _guard_clamp(y_mix: np.ndarray, idx_orig_set: set, best_true: float) -> np.ndarray:
        y = np.asarray(y_mix, dtype=float).copy()
        for i in range(y.size):
            if i not in idx_orig_set and y[i] < best_true:
                y[i] = best_true
        return y

    # ------------------------------------------------------------------ #
    # Main library API
    # ------------------------------------------------------------------ #

    def optimize(self, problem: Any, max_evals: int) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run DTS-CMA-ES on `problem` with an original-evaluation budget `max_evals`.
        Returns:
            best_x (truncated), best_f, history_best_f
        """
        print(f"sigma0 = {self.sigma0}")
        if cma is None:  # pragma: no cover
            raise ImportError(f"pycma (package 'cma') is required: {_cma_import_error}")

        if max_evals <= 0:
            return self._truncate_solution(np.zeros(self.dim)), float("inf"), []

        eval_batch = self._build_batch_evaluator(problem)

        # Bounds only used to sample x0 if user didn't pass x0
        lower_bound = getattr(problem, "lower_bound", None)
        upper_bound = getattr(problem, "upper_bound", None)
        if lower_bound is None or upper_bound is None:
            lower = np.full(self.dim, -1.0)
            upper = np.full(self.dim, 1.0)
        else:
            lower = np.asarray(lower_bound, dtype=float).reshape(-1)
            upper = np.asarray(upper_bound, dtype=float).reshape(-1)
            if lower.size == 1:
                lower = np.full(self.dim, float(lower))
            if upper.size == 1:
                upper = np.full(self.dim, float(upper))

        x0 = self.x0.copy()
        sigma0 = float(self.sigma0)

        # CMA-ES options
        opts = dict(self.cma_options)
        opts.setdefault("popsize", self._resolve_popsize())
        opts.setdefault("seed", self.seed)
        opts.setdefault("verb_log", 0)
        opts.setdefault("verb_disp", 0)  # we handle printing ourselves

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        print("CMA opts seed =", es.opts.get("seed", None))

        archive = _Archive(self.dim)
        model_cache: Optional[_ModelBundle] = None

        alpha = float(self.alpha0)
        if self.use_adaptive_alpha:
            alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
        else:
            alpha = float(np.clip(alpha, 0.0, 1.0))
        eps_smooth = 0.0

        history: List[float] = []
        eval_count = 0
        gen = 0

        # while not es.stop() and eval_count < max_evals:
        while eval_count < max_evals:
            gen += 1

            # Cache aging
            if model_cache is not None:
                model_cache.age += 1
                if model_cache.age > self.max_model_age:
                    model_cache = None

            # Ask
            X_pop_list = es.ask()
            X_pop = np.asarray(X_pop_list, dtype=float)
            lamb = int(X_pop.shape[0])

            mean = np.asarray(es.mean, dtype=float)
            sigma = float(es.sigma)
            C = np.asarray(getattr(es, "C", np.eye(self.dim)), dtype=float)

            remaining = max_evals - eval_count
            if remaining <= 0:
                break

            # Warm-up: evaluate entire population until archive has enough points
            if archive.size < self.warmup_min_points:
                if remaining < lamb:
                    break  # can't finish a full warmup generation without overshooting budget
                y_true = np.asarray(eval_batch(list(X_pop)), dtype=float).reshape(-1)
                archive.add_batch(X_pop, y_true)
                eval_count += lamb
                es.tell(X_pop_list, list(y_true))

                _, best_f = archive.best()
                history.append(best_f)
                if self.print_every > 0 and (gen % self.print_every == 0):
                    print(f"[DTS] iter {gen:4d} | true_evals {eval_count:6d} | best f = {best_f:.6e}")
                continue

            # ------------------------------------------------------------------
            # NEW: Only enable surrogate if there are at least n_min_train points
            #      within the Mahalanobis-radius (in whitened space) around mean.
            #      Otherwise, do a full true-evaluation generation.
            #      Controlled by require_min_points_in_radius.
            # ------------------------------------------------------------------
            if self.require_min_points_in_radius:
                X_arch, y_arch = archive.as_arrays()
                r_A_max = self._compute_r_A_max()
                X_arch_white = self._whiten(X_arch, mean, sigma, C)
                sq_norm = np.sum(X_arch_white ** 2, axis=1)
                n_inside = int(np.sum(sq_norm <= (r_A_max ** 2)))

                if n_inside < self.n_min_train:
                    if remaining < lamb:
                        break
                    y_true = np.asarray(eval_batch(list(X_pop)), dtype=float).reshape(-1)
                    archive.add_batch(X_pop, y_true)
                    eval_count += lamb
                    es.tell(X_pop_list, list(y_true))

                    _, best_f = archive.best()
                    history.append(best_f)
                    if self.print_every > 0 and (gen % self.print_every == 0):
                        print(f"[DTS] iter {gen:4d} | true_evals {eval_count:6d} | best f = {best_f:.6e}")
                    continue

            # Train GP1 (or fallback to cache)
            X_arch, y_arch = archive.as_arrays()
            bundle1 = self._train_gp_model(X_arch, y_arch, mean, sigma, C)
            bundle1_is_new = bundle1 is not None
            if bundle1 is None and self.use_model_cache and model_cache is not None:
                bundle1 = model_cache
                bundle1_is_new = False

            # If no model at all, fallback to full evaluation
            if bundle1 is None:
                if remaining < lamb:
                    break
                y_true = np.asarray(eval_batch(list(X_pop)), dtype=float).reshape(-1)
                archive.add_batch(X_pop, y_true)
                eval_count += lamb
                es.tell(X_pop_list, list(y_true))

                _, best_f = archive.best()
                history.append(best_f)
                if self.print_every > 0 and (gen % self.print_every == 0):
                    print(f"[DTS] iter {gen:4d} | true_evals {eval_count:6d} | best f = {best_f:.6e}")
                continue

            # GP1 predict
            Xw1 = self._whiten(X_pop, bundle1.mean, bundle1.sigma, bundle1.C)
            mu1, var1 = bundle1.gp.predict(Xw1)
            std1 = np.sqrt(np.maximum(var1, 1e-18))

            # Decide number of true evaluations this generation
            n_orig_target = int(math.ceil(alpha * lamb))
            n_orig_target = max(self.min_true_per_gen, n_orig_target)
            n_orig_target = min(n_orig_target, lamb)

            n_orig = min(n_orig_target, remaining)
            if n_orig <= 0:
                break

            # Select points via decoupled ranker
            idx_orig = self.ranker.select(
                mean=mu1,
                std=std1,
                y_train_min=bundle1.y_train_min,
                y_train_max=bundle1.y_train_max,
                n_select=n_orig,
            )
            idx_orig_set = set(int(i) for i in idx_orig)

            # Evaluate selected points
            X_orig = X_pop[idx_orig]
            y_orig = np.asarray(eval_batch(list(X_orig)), dtype=float).reshape(-1)
            archive.add_batch(X_orig, y_orig)
            eval_count += int(y_orig.size)

            # Train GP2 on updated archive (fallback to GP1 if fail)
            X_arch2, y_arch2 = archive.as_arrays()
            bundle2 = self._train_gp_model(X_arch2, y_arch2, mean, sigma, C)
            bundle2_is_new = bundle2 is not None
            skip_adapt = False
            if bundle2 is None:
                bundle2 = bundle1
                bundle2_is_new = False
                skip_adapt = True

            # GP2 predict & mix
            Xw2 = self._whiten(X_pop, bundle2.mean, bundle2.sigma, bundle2.C)
            mu2, var2 = bundle2.gp.predict(Xw2)
            y_mix = np.asarray(mu2, dtype=float).copy()

            for j, idx in enumerate(idx_orig):
                y_mix[int(idx)] = float(y_orig[j])

            # Prediction guard
            _, best_true = archive.best()
            if self.prediction_guard == "shift":
                y_mix = self._guard_shift(y_mix, idx_orig_set, best_true)
            elif self.prediction_guard == "clamp":
                y_mix = self._guard_clamp(y_mix, idx_orig_set, best_true)
            elif self.prediction_guard == "none":
                pass
            else:
                raise ValueError(f"Unknown prediction_guard: {self.prediction_guard}")

            # Optional alpha adaptation (Algorithm 4)
            if self.use_adaptive_alpha and not skip_adapt:
                mu_eff = getattr(es.sp, "mu", max(1, lamb // 2))
                eps_rde = _rde_mu(mu1, y_mix, mu=int(mu_eff))
                eps_smooth = (1.0 - self.beta) * eps_smooth + self.beta * eps_rde
                alpha = _update_alpha_self_adaptive(
                    eps_smooth=eps_smooth,
                    alpha_current=alpha,
                    dim=self.dim,
                    alpha_min=self.alpha_min,
                    alpha_max=self.alpha_max,
                    clamp_dim_to_20=True,
                )

            # CMA-ES update with mixed fitness values
            es.tell(X_pop_list, list(y_mix))

            # Update cache (prefer newest model2)
            if self.use_model_cache:
                if bundle2_is_new:
                    model_cache = bundle2
                    model_cache.age = 0
                elif bundle1_is_new:
                    model_cache = bundle1
                    model_cache.age = 0

            _, best_f = archive.best()
            history.append(best_f)
            if self.print_every > 0 and (gen % self.print_every == 0):
                print(f"[DTS] iter {gen:4d} | true_evals {eval_count:6d} | best f = {best_f:.6e}")

        best_x, best_f = archive.best()
        if best_x is None:
            best_x = np.zeros(self.dim, dtype=float)
            best_f = float("inf")

        # Save outputs for inspection
        self.stop_reasons_ = dict(es.stop())
        self.evals_ = int(eval_count)
        self.history_ = list(history)

        return self._truncate_solution(best_x), float(best_f), history


# =============================================================================
# Demo (library-style): run on AckleyProblem
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from problems.ackley import AckleyProblem  # type: ignore

    dim = 20
    max_evals = 1000

    problem = AckleyProblem(dim=dim)

    opt = DTSCMAESOptimizer(
        dim=dim,
        seed=2,
        sigma0=9.0,
        selection_criterion="cstd",
        prediction_guard="shift",
        print_every=500,
        # NEW: enable the local-only surrogate gate
        # require_min_points_in_radius=True,
        require_min_points_in_radius=False,
    )

    best_x, best_f, hist = opt.optimize(problem, max_evals=max_evals)

    print("\n=== DTS-CMA-ES (library refactor) on Ackley ===")
    print(f"best f: {best_f:.6e}")
    print(f"best x (first 10): {best_x}")
    print(f"true evals used: {opt.evals_}")
    print("stop reasons:", opt.stop_reasons_)
