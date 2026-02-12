#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTS-CMA-ES (Doubly Trained Surrogate CMA-ES) on top of pycma (ask–tell).

Modified version2:
  - robuster & closer to papers
  - ceil(alpha*lambda)
  - optional model caching
  - selection criterion switchable
  - prediction guard: shift/clamp/none
  - ALL tunables in DTSConfig / GPConfig

NOTE: Python 3.7 compatible (no statistics.NormalDist).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import cma  # pycma

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning


# =============================================================================
# Small numeric helpers
# =============================================================================

def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF Φ(z), vectorized, without scipy."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def normal_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF φ(z), vectorized."""
    z = np.asarray(z, dtype=float)
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z * z)


def norm_ppf(p: float) -> float:
    """
    Approximate inverse CDF (PPF) of standard normal distribution.

    Uses Peter J. Acklam's rational approximation.
    Works well for p in (0,1). No scipy, Python 3.7 compatible.
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")

    # Coefficients in rational approximations
    a = [-3.969683028665376e+01,
          2.209460984245205e+02,
         -2.759285104469687e+02,
          1.383577518672690e+02,
         -3.066479806614716e+01,
          2.506628277459239e+00]

    b = [-5.447609879822406e+01,
          1.615858368580409e+02,
         -1.556989798598866e+02,
          6.680131188771972e+01,
         -1.328068155288572e+01]

    c = [-7.784894002430293e-03,
         -3.223964580411365e-01,
         -2.400758277161838e+00,
         -2.549732539343734e+00,
          4.374664141464968e+00,
          2.938163982698783e+00]

    d = [ 7.784695709041462e-03,
          3.224671290700398e-01,
          2.445134137142996e+00,
          3.754408661907416e+00]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return -num / den

    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    return (num / den) * q


def chi2_ppf_wilson_hilferty(p: float, df: int) -> float:
    """
    Approximate chi-square quantile using Wilson–Hilferty transform.

    Returns Q such that P(Chi2(df) <= Q) ≈ p.
    """
    df = max(1, int(df))
    p = float(p)
    p = min(max(p, 1e-12), 1 - 1e-12)

    z = norm_ppf(p)  # <-- fixed: no NormalDist (Py3.7 compatible)
    a = 2.0 / (9.0 * df)
    q = df * (1.0 - a + z * math.sqrt(a)) ** 3
    return float(max(q, 1e-12))


def rde_mu(y1: np.ndarray, y2: np.ndarray, mu: int) -> float:
    """
    Ranking Difference Error RDE_mu(y1, y2) (Bajer et al., 2019, Eq. 14).
    y2 is assumed to be more accurate.

    Uses 0-based ranks. Denominator = mu * (lambda - mu).
    """
    y1 = np.asarray(y1, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()
    if y1.shape != y2.shape:
        raise ValueError("y1 and y2 must have the same shape")
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
    if denom <= 0.0:
        return 0.0
    return float(num / denom)


# =============================================================================
# Self-adaptive alpha (Algorithm 4 in Bajer et al. 2019)
# =============================================================================

def compute_eps_bounds(alpha: float, dim: int, clamp_dim_to_20: bool = True) -> Tuple[float, float]:
    """
    Compute epsilon_min, epsilon_max using the linear regression models Q2_min, Q3_max.
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


def update_alpha_self_adaptive(
    eps_smooth: float,
    alpha_current: float,
    dim: int,
    alpha_min: float,
    alpha_max: float,
    clamp_dim_to_20: bool = True,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> float:
    """Fixed-point iteration from the paper."""
    alpha = float(alpha_current)
    alpha = max(alpha_min, min(alpha_max, alpha))

    for _ in range(int(max_iter)):
        eps_min, eps_max = compute_eps_bounds(alpha, dim, clamp_dim_to_20=clamp_dim_to_20)
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
# Config (ALL tunable parameters are here)
# =============================================================================

Objective = Callable[[np.ndarray], float]


@dataclass
class GPConfig:
    nu: float = 2.5

    constant_value: float = 1.0
    constant_bounds: Tuple[float, float] = (1e-3, 1e3)

    length_scale: float = 1.0
    length_scale_bounds: Tuple[float, float] = (1e-2, 1e2)

    noise_level: float = 1e-6
    noise_bounds: Tuple[float, float] = (1e-10, 1e-3)

    n_restarts_optimizer: int = 1
    random_state: Optional[int] = None

    y_std_min: float = 1e-12


@dataclass
class DTSConfig:
    # CMA-ES settings
    popsize_mode: str = "double"     # "default" / "double" / "custom"
    popsize_custom: Optional[int] = None
    cma_options: Dict[str, Any] = field(default_factory=dict)

    # Surrogate usage
    alpha0: float = 0.05
    use_adaptive_alpha: bool = False
    beta: float = 0.3
    alpha_min: float = 0.04
    alpha_max: float = 1.0
    min_true_per_gen: int = 1

    # Warm-up / training sizes
    warmup_min_points: Optional[int] = None   # default: max(10*D, 50)
    n_min_train: Optional[int] = None         # default: max(5, 3*D)
    n_max_train: Optional[int] = None         # default: min(20*D, 300)

    # Training radius
    radius_mode: str = "chi2"                 # "chi2" or "sqrt_dim"
    radius_chi2_p: float = 0.99
    r_A_max_factor: float = 4.0

    # Selection criterion
    selection_criterion: str = "cstd"         # "cstd" / "cpoi" / "cei" / "mean"
    cpoi_eps: float = 0.05

    # Prediction guard
    prediction_guard: str = "shift"           # "shift" / "clamp" / "none"

    # Cache model
    use_model_cache: bool = True
    max_model_age: int = 2

    # Whitening jitter
    whiten_jitter: float = 1e-12

    # GP config
    gp: GPConfig = field(default_factory=GPConfig)


# =============================================================================
# Gaussian Process surrogate
# =============================================================================

class GaussianProcessSurrogate:
    """Wrapper around sklearn GaussianProcessRegressor. Expects whitened X."""

    def __init__(self, dim: int, cfg: GPConfig):
        self.dim = int(dim)
        self.cfg = cfg

        kernel = (
            ConstantKernel(cfg.constant_value, cfg.constant_bounds)
            * Matern(length_scale=cfg.length_scale, length_scale_bounds=cfg.length_scale_bounds, nu=cfg.nu)
            + WhiteKernel(noise_level=cfg.noise_level, noise_level_bounds=cfg.noise_bounds)
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=False,
            n_restarts_optimizer=int(cfg.n_restarts_optimizer),
            random_state=cfg.random_state,
        )

        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.is_trained_: bool = False

    def fit(self, Xw: np.ndarray, y: np.ndarray) -> bool:
        Xw = np.asarray(Xw, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if Xw.shape[0] != y.shape[0]:
            raise ValueError("X and y size mismatch")
        if Xw.shape[0] < 2:
            return False

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_
        y_std = float(np.std(y_centered))
        if y_std < self.cfg.y_std_min:
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
# Archive
# =============================================================================

class Archive:
    """Store all true evaluations."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    def add_one(self, x: np.ndarray, fx: float) -> None:
        self.X.append(np.asarray(x, dtype=float).copy())
        self.y.append(float(fx))

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.X:
            return np.empty((0, self.dim), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(self.X), np.asarray(self.y, dtype=float)

    def best(self) -> Tuple[Optional[np.ndarray], float]:
        if not self.y:
            return None, float("inf")
        i = int(np.argmin(self.y))
        return self.X[i].copy(), float(self.y[i])


# =============================================================================
# Model cache bundle
# =============================================================================

@dataclass
class ModelBundle:
    gp: GaussianProcessSurrogate
    mean: np.ndarray
    sigma: float
    C: np.ndarray
    y_train_min: float
    y_train_max: float
    age: int = 0


# =============================================================================
# DTS-CMA-ES Optimizer
# =============================================================================

class DtsCmaEsOptimizer:
    def __init__(
        self,
        f: Objective,
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
        self.cfg = config
        self.seed = seed

        self.dim = int(self.x0.size)

        # Dimension-dependent defaults
        self.warmup_min_points = int(config.warmup_min_points or max(10 * self.dim, 50))
        self.n_min_train = int(config.n_min_train or max(5, 3 * self.dim))
        self.n_max_train = int(config.n_max_train or min(20 * self.dim, 300))
        if self.n_max_train < self.n_min_train:
            self.n_max_train = self.n_min_train
        if self.warmup_min_points < self.n_min_train:
            self.warmup_min_points = self.n_min_train

        # Alpha
        if config.use_adaptive_alpha:
            self.alpha = float(np.clip(config.alpha0, config.alpha_min, config.alpha_max))
        else:
            self.alpha = float(np.clip(config.alpha0, 0.0, 1.0))
        self.eps_smooth = 0.0

        popsize = self._resolve_popsize(self.dim, config)

        opts = dict(config.cma_options) if config.cma_options else {}
        opts.setdefault("popsize", popsize)
        opts.setdefault("seed", seed)
        opts.setdefault("verb_log", 0)
        opts.setdefault("verb_disp", 5)

        self.es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, opts)

        self.archive = Archive(self.dim)
        self.model_cache: Optional[ModelBundle] = None

        self.eval_count = 0
        self.history_evals: List[int] = []
        self.history_best: List[float] = []

    @staticmethod
    def _resolve_popsize(dim: int, cfg: DTSConfig) -> int:
        d = max(2, int(dim))
        mode = str(cfg.popsize_mode).lower()
        if mode == "custom":
            if cfg.popsize_custom is None:
                raise ValueError("popsize_mode='custom' requires popsize_custom")
            return int(cfg.popsize_custom)
        if mode == "default":
            return 4 + int(math.floor(3.0 * math.log(d)))
        if mode == "double":
            return 8 + int(math.ceil(6.0 * math.log(d)))
        raise ValueError(f"Unknown popsize_mode: {cfg.popsize_mode}")

    def _compute_r_A_max(self) -> float:
        mode = str(self.cfg.radius_mode).lower()
        if mode == "sqrt_dim":
            return float(self.cfg.r_A_max_factor * math.sqrt(self.dim))
        if mode == "chi2":
            q = chi2_ppf_wilson_hilferty(self.cfg.radius_chi2_p, self.dim)
            return float(self.cfg.r_A_max_factor * math.sqrt(q))
        raise ValueError(f"Unknown radius_mode: {self.cfg.radius_mode}")

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
        cov = cov + float(self.cfg.whiten_jitter) * np.eye(self.dim)

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, float(self.cfg.whiten_jitter), None)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        Z = np.linalg.solve(L, dx.T).T
        return Z

    def _select_training_indices(self, X_white: np.ndarray, r_A_max: float) -> np.ndarray:
        n_total = int(X_white.shape[0])
        if n_total == 0:
            return np.empty((0,), dtype=int)

        sq_norm = np.sum(X_white ** 2, axis=1)
        inside = np.where(sq_norm <= (r_A_max ** 2))[0]

        if inside.size < self.n_min_train:
            k = min(self.n_max_train, n_total)
            idx = np.argsort(sq_norm)[:k]
            return idx.astype(int)

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
    ) -> Optional[ModelBundle]:
        if X_archive.shape[0] < self.n_min_train:
            return None

        X_white = self._whiten(X_archive, mean, sigma, C)
        r_A_max = self._compute_r_A_max()
        idx = self._select_training_indices(X_white, r_A_max)

        if idx.size < self.n_min_train:
            return None

        X_train = X_white[idx]
        y_train = y_archive[idx]

        gp = GaussianProcessSurrogate(dim=self.dim, cfg=self.cfg.gp)
        ok = gp.fit(X_train, y_train)
        if not ok:
            return None

        return ModelBundle(
            gp=gp,
            mean=np.asarray(mean, dtype=float).copy(),
            sigma=float(sigma),
            C=np.asarray(C, dtype=float).copy(),
            y_train_min=float(np.min(y_train)),
            y_train_max=float(np.max(y_train)),
            age=0,
        )

    def _select_original_indices(
        self,
        mu: np.ndarray,
        std: np.ndarray,
        y_train_min: float,
        y_train_max: float,
        n_orig: int,
    ) -> np.ndarray:
        mu = np.asarray(mu, dtype=float).ravel()
        std = np.asarray(std, dtype=float).ravel()
        lamb = int(mu.size)
        n_orig = max(1, min(int(n_orig), lamb))

        crit = str(self.cfg.selection_criterion).lower()
        if crit == "cstd":
            score = std
        elif crit == "mean":
            score = -mu
        elif crit == "cpoi":
            T = y_train_min - float(self.cfg.cpoi_eps) * (y_train_max - y_train_min)
            z = (T - mu) / (std + 1e-18)
            score = normal_cdf(z)
        elif crit == "cei":
            y_best = y_train_min
            z = (y_best - mu) / (std + 1e-18)
            score = (y_best - mu) * normal_cdf(z) + std * normal_pdf(z)
        else:
            raise ValueError(f"Unknown selection_criterion: {self.cfg.selection_criterion}")

        idx_sorted = np.argsort(-score)
        return idx_sorted[:n_orig].astype(int)

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

    def run(self) -> Tuple[np.ndarray, float]:
        es = self.es
        cfg = self.cfg

        while not es.stop() and self.eval_count < self.max_evals:
            if self.model_cache is not None:
                self.model_cache.age += 1
                if self.model_cache.age > int(cfg.max_model_age):
                    self.model_cache = None

            X_pop = np.asarray(es.ask())
            lamb = int(X_pop.shape[0])

            mean = np.asarray(es.mean)
            sigma = float(es.sigma)
            C = np.asarray(getattr(es, "C", np.eye(self.dim)))

            X_arch, y_arch = self.archive.as_arrays()
            if X_arch.shape[0] < self.warmup_min_points:
                y_true = [self._eval_and_record(x) for x in X_pop]
                es.tell(X_pop, y_true)
                self._log_iteration()
                continue

            bundle1 = self._train_gp_model(X_arch, y_arch, mean, sigma, C)
            bundle1_is_new = bundle1 is not None
            if bundle1 is None and cfg.use_model_cache and self.model_cache is not None:
                bundle1 = self.model_cache
                bundle1_is_new = False

            if bundle1 is None:
                y_true = [self._eval_and_record(x) for x in X_pop]
                es.tell(X_pop, y_true)
                self._log_iteration()
                continue

            Xw1 = self._whiten(X_pop, bundle1.mean, bundle1.sigma, bundle1.C)
            mu1, var1 = bundle1.gp.predict(Xw1)
            std1 = np.sqrt(np.maximum(var1, 1e-18))

            n_orig_target = int(math.ceil(self.alpha * lamb))
            n_orig_target = max(int(cfg.min_true_per_gen), n_orig_target)
            n_orig_target = min(n_orig_target, lamb)

            remaining = self.max_evals - self.eval_count
            if remaining <= 0:
                break
            n_orig = min(n_orig_target, remaining)
            if n_orig <= 0:
                break

            idx_orig = self._select_original_indices(mu1, std1, bundle1.y_train_min, bundle1.y_train_max, n_orig)
            idx_orig_set = set(int(i) for i in idx_orig)

            y_orig: List[float] = []
            for i in idx_orig:
                y_orig.append(self._eval_and_record(X_pop[int(i)]))

            X_arch2, y_arch2 = self.archive.as_arrays()
            bundle2 = self._train_gp_model(X_arch2, y_arch2, mean, sigma, C)
            bundle2_is_new = bundle2 is not None
            skip_adapt = False
            if bundle2 is None:
                bundle2 = bundle1
                bundle2_is_new = False
                skip_adapt = True

            Xw2 = self._whiten(X_pop, bundle2.mean, bundle2.sigma, bundle2.C)
            mu2, var2 = bundle2.gp.predict(Xw2)
            y_mix = np.asarray(mu2, dtype=float).copy()

            for j, idx in enumerate(idx_orig):
                y_mix[int(idx)] = float(y_orig[j])

            _, best_true = self.archive.best()
            guard = str(cfg.prediction_guard).lower()
            if guard == "shift":
                y_mix = self._guard_shift(y_mix, idx_orig_set, best_true)
            elif guard == "clamp":
                y_mix = self._guard_clamp(y_mix, idx_orig_set, best_true)
            elif guard == "none":
                pass
            else:
                raise ValueError(f"Unknown prediction_guard: {cfg.prediction_guard}")

            if cfg.use_adaptive_alpha and not skip_adapt:
                mu_eff = getattr(es.sp, "mu", max(1, lamb // 2))
                eps_rde = rde_mu(mu1, y_mix, mu=int(mu_eff))
                self.eps_smooth = (1.0 - cfg.beta) * self.eps_smooth + cfg.beta * eps_rde
                self.alpha = update_alpha_self_adaptive(
                    eps_smooth=self.eps_smooth,
                    alpha_current=self.alpha,
                    dim=self.dim,
                    alpha_min=cfg.alpha_min,
                    alpha_max=cfg.alpha_max,
                    clamp_dim_to_20=True,
                )

            es.tell(X_pop, list(y_mix))
            self._log_iteration()

            if cfg.use_model_cache:
                if bundle2_is_new:
                    self.model_cache = bundle2
                    self.model_cache.age = 0
                elif bundle1_is_new:
                    self.model_cache = bundle1
                    self.model_cache.age = 0

        best_x, best_f = self.archive.best()
        if best_x is None:
            best_x = np.asarray(self.es.best.x)
            best_f = float(self.es.best.f)
        return best_x, best_f

    def _eval_and_record(self, x: np.ndarray) -> float:
        fx = float(self.f(np.asarray(x, dtype=float)))
        self.eval_count += 1
        self.archive.add_one(x, fx)
        return fx

    def _log_iteration(self) -> None:
        _, best_f = self.archive.best()
        self.history_evals.append(self.eval_count)
        self.history_best.append(best_f)

        try:
            disp_gap = int(self.es.opts.get("verb_disp", 0))
        except Exception:
            disp_gap = 0
        if disp_gap > 0 and (len(self.history_best) % disp_gap == 0):
            print(f"[DTS] iter {len(self.history_best):4d} | true_evals {self.eval_count:6d} | best f = {best_f:.6e}")


# =============================================================================
# Example benchmark: Ackley
# =============================================================================

# def ackley(x: np.ndarray) -> float:
#     """Non-shifted Ackley (kept for reference)."""
#     x = np.asarray(x, dtype=float)
#     d = x.size
#     a, b, c = 20.0, 0.2, 2 * math.pi
#     sum_sq = float(np.dot(x, x))
#     sum_cos = float(np.sum(np.cos(c * x)))
#     term1 = -a * math.exp(-b * math.sqrt(sum_sq / d))
#     term2 = -math.exp(sum_cos / d)
#     return float(term1 + term2 + a + math.e)

# Use shifted/rotated Ackley from problems.ackley (vectorized, stored optimum)
from problems.ackley import AckleyProblem  # type: ignore


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    dim = 20
    max_evals = 1000

    rng = np.random.RandomState(52)
    problem = AckleyProblem(dim=dim)
    x0 = rng.uniform(problem.lower_bound, problem.upper_bound, size=dim)
    sigma0 = 9.0

    # ==================== ALL TUNABLE PARAMETERS HERE ====================
    cfg = DTSConfig(
        popsize_mode="double",
        cma_options={
            "seed": 5,
            "verb_disp": 10,
        },

        alpha0=0.05,
        use_adaptive_alpha=False,
        beta=0.3,
        alpha_min=0.04,
        alpha_max=1.0,
        min_true_per_gen=1,

        warmup_min_points=max(20 * dim, 50),
        n_min_train=max(5, 3 * dim),
        n_max_train=min(20 * dim, 300),

        radius_mode="chi2",
        radius_chi2_p=0.99,
        r_A_max_factor=4.0,

        selection_criterion="cstd",
        cpoi_eps=0.05,

        prediction_guard="shift",

        use_model_cache=True,
        max_model_age=2,

        whiten_jitter=1e-12,

        gp=GPConfig(
            nu=2.5,
            constant_bounds=(1e-3, 1e3),
            length_scale_bounds=(1e-2, 1e2),
            noise_bounds=(1e-10, 1e-3),
            n_restarts_optimizer=1,
            random_state=123,
        ),
    )
    # =====================================================================

    opt = DtsCmaEsOptimizer(
        f=lambda x: float(np.ravel(problem.evaluate(np.asarray(x, dtype=float)))[0]),
        x0=x0,
        sigma0=sigma0,
        max_evals=max_evals,
        config=cfg,
        seed=1,
    )
    best_x, best_f = opt.run()

    print("\n=== DTS-CMA-ES on Ackley ===")
    print(f"best f: {best_f:.6e}")
    print(f"best x (first 5): {best_x[:5]}")
    print(f"true evals used: {opt.eval_count}")
    print("CMA stop reasons:", opt.es.stop())


if __name__ == "__main__":
    main()
