#!/usr/bin/env python3
"""
DTS-CMA-ES on top of pycma (ask–tell interface), with adaptive α.

Implements the doubly trained surrogate evolution control from:
- Pitra et al., "Doubly Trained Evolution Control for the Surrogate CMA-ES",
  PPSN 2016.:contentReference[oaicite:14]{index=14}
- Bajer et al., "Gaussian Process Surrogate Models for CMA-ES",
  Evolutionary Computation Journal 2019.:contentReference[oaicite:15]{index=15}

Key features of this implementation
-----------------------------------
- CMA-ES sampling and parameter update are delegated to pycma (CMAEvolutionStrategy).
- A Gaussian Process surrogate (scikit-learn) is trained on an archive of
  original evaluations.
- Doubly trained evolution control (DTS):
  * Train GP1 on archive, predict current population.
  * Select ceil(alpha * lambda) points for original evaluation using PoI.
  * Retrain GP2 on updated archive.
  * Predict the rest of the population; replace predicted values for
    originally evaluated solutions with their true fitness.
- Self-adaptation of alpha based on a smoothed Ranking Difference Error RDE_mu
  between the two model predictions, using the regression-based transfer
  function from Pitra et al. (2017).
- Designed to be reasonably efficient in 100 dimensions by capping the GP
  training set size.

Dependencies
------------
- numpy
- pycma (import as `cma`)
- scikit-learn (for GaussianProcessRegressor)
- (optional) matplotlib for plotting (not required to run the optimization)

Usage
-----
$ python dts_cma_es.py

This will:
- Run DTS-CMA-ES on a 100D Rastrigin function.
- Run plain CMA-ES as a baseline with comparable settings.
- Print convergence information and final results.

You can edit `main()` at the bottom to change the benchmark, dimensionality,
budget, etc.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
from scipy.special import erf

import numpy as np

try:
    import cma  # pycma library
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "This script requires the `cma` (pycma) library. "
        "Install it with `pip install cma`."
    ) from e

# Surrogate model: Gaussian process via scikit-learn
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
except ImportError:
    GaussianProcessRegressor = None  # we will check later and fall back gracefully


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def standard_normal_cdf(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))

def compute_eps_bounds(alpha: float, dim: int) -> Tuple[float, float]:
    """
    Regression-based bounds eps_min, eps_max used in the α self-adaptation
    (Algorithm 4 in the ECJ paper).

    eps_min = (1, ln D, α, α ln D, α^2) · b_min
    eps_max = (1, ln D, α, α ln D, α^2) · b_max

    with
        b_min  = (0.11, -0.0092, -0.13, 0.044, 0.14)
        b_max  = (0.35, -0.047, 0.44, 0.044, -0.19)

    We clamp results to [0, 1] and ensure eps_max > eps_min.
    """
    assert dim >= 1
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.5:
        # sanity check; for our purposes alpha should be in [0,1]
        alpha = max(0.0, min(1.0, alpha))

    lnD = math.log(dim)
    v = np.array([1.0, lnD, alpha, alpha * lnD, alpha * alpha])

    b_min = np.array([0.11, -0.0092, -0.13, 0.044, 0.14])
    b_max = np.array([0.35, -0.047, 0.44, 0.044, -0.19])

    eps_min = float(v @ b_min)
    eps_max = float(v @ b_max)

    # Clamp to [0, 1] and ensure ordering
    eps_min = max(0.0, min(1.0, eps_min))
    eps_max = max(0.0, min(1.0, eps_max))
    if eps_max <= eps_min:
        eps_max = eps_min + 1e-12
    return eps_min, eps_max


def ranking_difference_error_mu(
    y_hat: np.ndarray, y_ref: np.ndarray, mu: int
) -> float:
    """
    Approximate RDE_mu(y_hat, y_ref) in [0, 1].

    y_ref is considered the more accurate vector (second model predictions with
    original-evaluated points replaced), y_hat is the first model prediction.

    Implementation:
    - Compute ranks r_hat, r_ref (1 = best).
    - Consider indices of the mu best points according to y_ref.
    - For each such index i, add |r_ref[i] - r_hat[i]|.
    - Normalize by mu * (lambda - mu), which corresponds to the case where
      the mu best under y_ref are placed entirely among the lambda - mu worst
      ranks under y_hat.

    This is simpler than the exact combinatorial normalization in Eq. (14),
    but qualitatively equivalent and keeps RDE_mu in [0,1].
    """
    y_hat = np.asarray(y_hat).ravel()
    y_ref = np.asarray(y_ref).ravel()
    assert y_hat.shape == y_ref.shape
    lam = y_hat.size
    assert 1 <= mu < lam

    # Ranks: 1 = best (smallest value)
    order_ref = np.argsort(y_ref)
    order_hat = np.argsort(y_hat)
    r_ref = np.empty(lam, dtype=int)
    r_hat = np.empty(lam, dtype=int)
    r_ref[order_ref] = np.arange(1, lam + 1)
    r_hat[order_hat] = np.arange(1, lam + 1)

    # Indices of mu best according to y_ref
    best_ref_indices = order_ref[:mu]
    num = 0.0
    for idx in best_ref_indices:
        num += abs(float(r_ref[idx]) - float(r_hat[idx]))

    denom = float(mu * (lam - mu))
    if denom <= 0:
        return 0.0
    return float(num / denom)


def select_training_points_tss2(
    archive_X: np.ndarray,
    archive_y: np.ndarray,
    pop_X: np.ndarray,
    mean: np.ndarray,
    sigma: float,
    r_max: float,
    n_max: int,
    n_min: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Approximate TSS2 (training set selection 2) from the ECJ paper:

    - Start from all archive points.
    - Optionally restrict to those within radius r_max (measured in units of σ).
    - For each archive point, compute its minimal distance to any current
      population point (again scaled by σ).
    - Select up to n_max archive points with smallest such distance.

    Parameters
    ----------
    archive_X, archive_y : arrays of shape (N, dim) and (N,)
    pop_X : array of shape (lambda, dim)
    mean : current CMA-ES mean (dim,)
    sigma : current CMA-ES step-size
    r_max : maximum radius (in σ-scaled Euclidean distance) from mean
    n_max : maximum training set size
    n_min : minimum training set size; if fewer are available we return None
    """
    if archive_X.size == 0 or archive_X.shape[0] < n_min:
        return None

    sigma_safe = max(float(sigma), 1e-12)

    # Distance from mean (for radius filter)
    d_mean = np.linalg.norm(archive_X - mean, axis=1) / sigma_safe
    mask = d_mean <= float(r_max)
    candidate_idx = np.nonzero(mask)[0]
    if candidate_idx.size < n_min:
        # If radius is too strict early on, ignore it and use all archive points.
        candidate_idx = np.arange(archive_X.shape[0])

    if candidate_idx.size < n_min:
        return None

    cand_X = archive_X[candidate_idx]
    cand_y = archive_y[candidate_idx]

    # Minimal distance from each candidate to any point in current population
    # (scaled by sigma)
    diff = cand_X[:, None, :] - pop_X[None, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)  # shape (Nc, lambda)
    min_dists = np.sqrt(np.min(sq_dists, axis=1)) / sigma_safe  # (Nc,)

    # Sort by distance and take up to n_max closest points
    order = np.argsort(min_dists)
    k = min(n_max, order.size)
    if k < n_min:
        return None

    chosen = order[:k]
    return cand_X[chosen], cand_y[chosen]


# ---------------------------------------------------------------------------
# Gaussian process surrogate wrapper
# ---------------------------------------------------------------------------

class SimpleGaussianProcess:
    """
    Lightweight wrapper around sklearn's GaussianProcessRegressor.

    - Uses Matérn(ν=2.5) kernel with automatic relevance determination (ARD).
    - Standardizes inputs (per-dimension mean/std) and outputs (mean/std).
    - Provides predict(X, return_std=True) -> (mean, std).

    In the original DTS-CMA-ES Matlab implementation, GPML is used with
    Matérn 5/2, zero mean, and hyperparameters fitted via ML; this class
    mirrors that setup in scikit-learn.
    """

    def __init__(self, dim: int):
        if GaussianProcessRegressor is None:
            raise RuntimeError(
                "scikit-learn is required for the GP surrogate. "
                "Install it with `pip install scikit-learn`."
            )

        self.dim = int(dim)

        # ARD Matérn nu=2.5 with separate length scales per dimension
        base_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(self.dim),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        noise_kernel = WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-1))

        self._gp = GaussianProcessRegressor(
            kernel=base_kernel + noise_kernel,
            alpha=0.0,
            normalize_y=False,  # we standardize manually
            n_restarts_optimizer=0,  # keep cost reasonable
        )

        # Standardization parameters
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        assert X.shape[0] == y.size

        # Input standardization (per dimension)
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        # avoid division by zero
        self.x_std[self.x_std < 1e-12] = 1.0

        Xs = (X - self.x_mean) / self.x_std

        # Output standardization
        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        if self.y_std < 1e-12:
            self.y_std = 1.0

        ys = (y - self.y_mean) / self.y_std

        self._gp.fit(Xs, ys)

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.asarray(X, dtype=float)
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("GP has not been fitted yet")

        Xs = (X - self.x_mean) / self.x_std
        if return_std:
            ys_mean, ys_std = self._gp.predict(Xs, return_std=True)
            ys_mean = ys_mean * self.y_std + self.y_mean
            ys_std = ys_std * self.y_std
            return ys_mean, ys_std
        else:
            ys_mean = self._gp.predict(Xs, return_std=False)
            ys_mean = ys_mean * self.y_std + self.y_mean
            return ys_mean, None


# ---------------------------------------------------------------------------
# DTS-CMA-ES wrapper on top of pycma
# ---------------------------------------------------------------------------

@dataclass
class DtsCmaEsResult:
    best_x: np.ndarray
    best_y: float
    evals: int  # number of true function evaluations
    alpha_final: float
    history_evals: np.ndarray
    history_best: np.ndarray
    alpha_history: np.ndarray
    es_result: Any  # pycma's es.result object


@dataclass
class DtsCmaEs:
    """
    DTS-CMA-ES wrapper using pycma.CMAEvolutionStrategy and a GP surrogate.

    Parameters
    ----------
    f : callable
        Objective function to minimize, f(x: np.ndarray) -> float.
    x0 : array-like
        Initial mean of CMA-ES.
    sigma0 : float
        Initial step-size.
    alpha0 : float
        Initial ratio of original-evaluated points (Algorithm 3, α^(0)).
    adaptive_alpha : bool
        If True, use self-adaptation (Algorithm 4). If False, keep α constant.
    dim : int
        Dimension of the search space.
    popsize_factor : float
        Factor to multiply CMA-ES default population size (λ). The DTS paper
        reports that using 2× population size often improves performance, so
        the default here is 2.0.
    n_max : int
        Maximum GP training set size (N_max). For high dimensions we cap at
        a moderate value for computational reasons.
    n_min : int
        Minimum GP training set size (N_min). If fewer archive points are
        available, we fall back to full original evaluation.
    beta : float
        Exponential smoothing rate for RDE (Algorithm 4).
    alpha_min, alpha_max : float
        Bounds on α used in the self-adaptation transfer function.
    r_max : float
        Maximum radius for training set selection (in units of σ).
    """

    f: Callable[[np.ndarray], float]
    x0: np.ndarray
    sigma0: float
    dim: int
    alpha0: float = 0.05
    adaptive_alpha: bool = True
    popsize_factor: float = 2.0
    n_max: Optional[int] = None
    n_min: Optional[int] = None
    beta: float = 0.3
    alpha_min: float = 0.04
    alpha_max: float = 1.0
    r_max: Optional[float] = None
    verbose: bool = True

    # Internal state (initialized in optimize)
    _archive_X: List[np.ndarray] = None
    _archive_y: List[float] = None

    def __post_init__(self):
        self.x0 = np.asarray(self.x0, dtype=float)
        assert self.x0.ndim == 1
        assert self.x0.size == self.dim

        if self.n_max is None:
            # default from ECJ paper: Nmax = 20 * D, but cap for high D
            self.n_max = min(20 * self.dim, 400)
        if self.n_min is None:
            self.n_min = max(3 * self.dim, 20)
        if self.r_max is None:
            # approximate 4 * sqrt(Q_chi^2(0.99, D)) by 4 * sqrt(D)
            self.r_max = 4.0 * math.sqrt(self.dim)

        self._archive_X = []
        self._archive_y = []

    # ---- Core optimization loop -------------------------------------------

    def optimize(
        self,
        max_evals: int,
        seed: Optional[int] = None,
        disp_gap_evals: int = 2000,
    ) -> DtsCmaEsResult:
        """Run DTS-CMA-ES until max_evals true evaluations or CMA-ES stops."""
        if GaussianProcessRegressor is None:
            if self.verbose:
                print(
                    "Warning: scikit-learn not available, "
                    "falling back to plain CMA-ES (no surrogates).",
                    file=sys.stderr,
                )
            # In this case we simply run plain CMA-ES with ask-tell, counting evals.
            return self._optimize_plain_cma(max_evals, seed, disp_gap_evals)

        # CMA-ES setup
        base_popsize = 4 + int(3 * math.log(self.dim))
        popsize = int(self.popsize_factor * base_popsize)
        options = {
            "popsize": popsize,
            "seed": seed,
            "verbose": -9,     # silence internal prints
            "verb_disp": 10,
            "verb_log": 0,
            "maxiter": 10**9,  # we control termination by max_evals
            "ftarget": -1e300, # effectively no ftarget
        }
        es = cma.CMAEvolutionStrategy(self.x0.tolist(), self.sigma0, options)

        alpha = float(self.alpha0)
        eps = 0.0  # smoothed error ε^(0); start at zero (perfect model assumption)

        evals = 0  # number of true f evaluations
        best_y = float("inf")
        best_x = self.x0.copy()

        history_evals: List[int] = []
        history_best: List[float] = []
        alpha_hist: List[float] = []

        last_disp_evals = 0

        generation = 0
        while not es.stop():
            generation += 1
            # Stop if budget exhausted
            if evals >= max_evals:
                break

            X = np.array(es.ask(), dtype=float)
            lam = X.shape[0]
            mu = lam // 2

            # If we don't yet have enough archive points, evaluate the full population
            if len(self._archive_X) < self.n_min:
                y = np.array([self.f(x) for x in X], dtype=float)
                for x_i, y_i in zip(X, y):
                    self._archive_X.append(np.array(x_i, copy=True))
                    self._archive_y.append(float(y_i))
                evals += lam
                y_for_cma = y
            else:
                # Surrogate is available: apply doubly trained evolution control
                archive_X = np.array(self._archive_X, dtype=float)
                archive_y = np.array(self._archive_y, dtype=float)

                # ---- First model: train GP1 on archive -------------------
                train1 = select_training_points_tss2(
                    archive_X,
                    archive_y,
                    pop_X=X,
                    mean=np.array(es.mean, dtype=float),
                    sigma=float(es.sigma),
                    r_max=self.r_max,
                    n_max=self.n_max,
                    n_min=self.n_min,
                )
                if train1 is None:
                    # Fallback: training set selection failed (e.g. too few points)
                    y = np.array([self.f(x) for x in X], dtype=float)
                    for x_i, y_i in zip(X, y):
                        self._archive_X.append(np.array(x_i, copy=True))
                        self._archive_y.append(float(y_i))
                    evals += lam
                    y_for_cma = y
                else:
                    X_tr1, y_tr1 = train1
                    gp1 = SimpleGaussianProcess(dim=self.dim)
                    try:
                        gp1.fit(X_tr1, y_tr1)
                    except Exception as e:
                        # If GP training fails, fall back to full evaluation for this gen
                        if self.verbose:
                            print(
                                f"[DTS] GP1 training failed in gen {generation}: {e}",
                                file=sys.stderr,
                            )
                        y = np.array([self.f(x) for x in X], dtype=float)
                        for x_i, y_i in zip(X, y):
                            self._archive_X.append(np.array(x_i, copy=True))
                            self._archive_y.append(float(y_i))
                        evals += lam
                        y_for_cma = y
                    else:
                        # Predict population with GP1
                        y_hat, s_hat = gp1.predict(X, return_std=True)
                        s_hat = np.maximum(s_hat, 1e-12)

                        # ---- Select points for original evaluation using PoI ----
                        archive_y_arr = np.array(self._archive_y, dtype=float)
                        if archive_y_arr.size == 0 or not np.isfinite(archive_y_arr).all():
                            # fallback: use standard deviation criterion if archive lacks info
                            criterion = s_hat
                        else:
                            y_min = float(np.min(archive_y_arr))
                            y_max = float(np.max(archive_y_arr))
                            if not np.isfinite(y_max) or y_max == y_min:
                                T = y_min
                            else:
                                # recommended threshold: T = ymin - 0.05 (ymax - ymin)
                                T = y_min - 0.05 * (y_max - y_min)
                            z = (T - y_hat) / s_hat
                            criterion = standard_normal_cdf(z)  # PoI

                        n_orig = max(1, int(round(alpha * lam)))
                        n_orig = min(lam, n_orig)
                        # Indices of the n_orig largest values of criterion
                        orig_indices = np.argsort(criterion)[-n_orig:]

                        # Evaluate selected points with true fitness
                        y_pred2 = np.empty(lam, dtype=float)
                        y_pred2[:] = np.nan
                        y_orig = {}

                        for idx in orig_indices:
                            fx = float(self.f(X[idx]))
                            y_orig[idx] = fx
                            self._archive_X.append(np.array(X[idx], copy=True))
                            self._archive_y.append(fx)
                        evals += len(orig_indices)

                        # ---- Second model: train GP2 on updated archive -------
                        archive_X2 = np.array(self._archive_X, dtype=float)
                        archive_y2 = np.array(self._archive_y, dtype=float)

                        train2 = select_training_points_tss2(
                            archive_X2,
                            archive_y2,
                            pop_X=X,
                            mean=np.array(es.mean, dtype=float),
                            sigma=float(es.sigma),
                            r_max=self.r_max,
                            n_max=self.n_max,
                            n_min=self.n_min,
                        )

                        if train2 is None:
                            # If training set selection fails, reuse GP1 predictions
                            y2, _ = y_hat, s_hat
                        else:
                            X_tr2, y_tr2 = train2
                            gp2 = SimpleGaussianProcess(dim=self.dim)
                            try:
                                gp2.fit(X_tr2, y_tr2)
                                y2, _ = gp2.predict(X, return_std=False), None
                            except Exception as e:
                                if self.verbose:
                                    print(
                                        f"[DTS] GP2 training failed in gen {generation}: {e}",
                                        file=sys.stderr,
                                    )
                                y2, _ = y_hat, s_hat

                        # Replace predictions for original-evaluated points with true values
                        y_for_cma = np.array(y2, copy=True)
                        for idx, fx in y_orig.items():
                            y_for_cma[idx] = fx

                        # ---- Self-adaptation of alpha -------------------------
                        if self.adaptive_alpha:
                            rde = ranking_difference_error_mu(y_hat, y_for_cma, mu=mu)
                            eps = (1.0 - self.beta) * eps + self.beta * rde
                            eps_min, eps_max = compute_eps_bounds(alpha, self.dim)
                            denom = max(eps_max - eps_min, 1e-12)
                            scaled = (eps - eps_min) / denom
                            if scaled < 0.0:
                                scaled = 0.0
                            elif scaled > 1.0:
                                scaled = 1.0
                            alpha = self.alpha_min + scaled * (self.alpha_max - self.alpha_min)

                        # If not adaptive, alpha stays constant

            # ---- CMA-ES update -------------------------------------------
            # CMA-ES uses only ranks of y_for_cma, so surrogate bias is acceptable.
            es.tell(X.tolist(), y_for_cma.tolist())

            # Track best (based on true evaluations only iff desired; here we
            # track best according to the y_for_cma that CMA-ES sees)
            idx_best = int(np.argmin(y_for_cma))
            if y_for_cma[idx_best] < best_y:
                best_y = float(y_for_cma[idx_best])
                best_x = X[idx_best].copy()

            history_evals.append(evals)
            history_best.append(best_y)
            alpha_hist.append(alpha)

            if self.verbose and evals - last_disp_evals >= disp_gap_evals:
                print(
                    f"[DTS] evals={evals:6d}, gen={generation:4d}, "
                    f"best_f={best_y:.4e}, alpha={alpha:.3f}"
                )
                last_disp_evals = evals

            if evals >= max_evals:
                break

        if self.verbose:
            print("[DTS] Terminated. es.stop() =", es.stop())

        res = es.result  # pycma's result object
        return DtsCmaEsResult(
            best_x=best_x,
            best_y=float(best_y),
            evals=evals,
            alpha_final=float(alpha),
            history_evals=np.array(history_evals, dtype=float),
            history_best=np.array(history_best, dtype=float),
            alpha_history=np.array(alpha_hist, dtype=float),
            es_result=res,
        )

    # ---- Plain CMA-ES fallback (no surrogate) -----------------------------

    def _optimize_plain_cma(
        self,
        max_evals: int,
        seed: Optional[int],
        disp_gap_evals: int,
    ) -> DtsCmaEsResult:
        """Run plain CMA-ES using the same popsize/budget, used if sklearn is missing."""
        base_popsize = 4 + int(3 * math.log(self.dim))
        popsize = int(self.popsize_factor * base_popsize)
        options = {
            "popsize": popsize,
            "seed": seed,
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "maxiter": 10**9,
            "ftarget": -1e300,
        }
        es = cma.CMAEvolutionStrategy(self.x0.tolist(), self.sigma0, options)

        evals = 0
        best_y = float("inf")
        best_x = self.x0.copy()

        history_evals: List[int] = []
        history_best: List[float] = []
        alpha_hist: List[float] = []

        last_disp_evals = 0
        generation = 0
        while not es.stop():
            generation += 1
            if evals >= max_evals:
                break

            X = np.array(es.ask(), dtype=float)
            lam = X.shape[0]
            y = np.array([self.f(x) for x in X], dtype=float)
            evals += lam

            es.tell(X.tolist(), y.tolist())

            idx_best = int(np.argmin(y))
            if y[idx_best] < best_y:
                best_y = float(y[idx_best])
                best_x = X[idx_best].copy()

            history_evals.append(evals)
            history_best.append(best_y)
            alpha_hist.append(1.0)  # effectively α = 1.0 (all original)

            if self.verbose and evals - last_disp_evals >= disp_gap_evals:
                print(
                    f"[CMA] evals={evals:6d}, gen={generation:4d}, "
                    f"best_f={best_y:.4e}"
                )
                last_disp_evals = evals

            if evals >= max_evals:
                break

        if self.verbose:
            print("[CMA] Terminated. es.stop() =", es.stop())

        res = es.result
        return DtsCmaEsResult(
            best_x=best_x,
            best_y=float(best_y),
            evals=evals,
            alpha_final=1.0,
            history_evals=np.array(history_evals, dtype=float),
            history_best=np.array(history_best, dtype=float),
            alpha_history=np.array(alpha_hist, dtype=float),
            es_result=res,
        )


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    """Sphere function: sum(x_i^2), optimum at x=0."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function (minimization, global optimum at x=0).

    f(x) = 10 D + sum_i (x_i^2 - 10 cos(2π x_i))
    """
    x = np.asarray(x, dtype=float)
    D = x.size
    return float(10.0 * D + np.sum(x ** 2 - 10.0 * np.cos(2.0 * math.pi * x)))


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment_100d():
    """Run DTS-CMA-ES and plain CMA-ES on a 100-dimensional benchmark."""
    dim = 100
    # You can switch between sphere and rastrigin here:
    objective = rastrigin
    # objective = sphere

    # Initial mean: start moderately far from optimum
    x0 = np.full(dim, 5.0, dtype=float)
    sigma0 = 3.0

    max_evals = 20000  # e.g., 200 evaluations per dimension
    seed = 42

    print("=== DTS-CMA-ES (adaptive alpha) on 100D Rastrigin ===")
    dts = DtsCmaEs(
        f=objective,
        x0=x0,
        sigma0=sigma0,
        dim=dim,
        alpha0=0.05,        # initial ratio (≈ 5% of population)
        adaptive_alpha=False,
        popsize_factor=2.0, # use 2× default popsize, as in DTS 2pop variants
        verbose=True,
    )
    dts_res = dts.optimize(max_evals=max_evals, seed=seed, disp_gap_evals=2000)

    print("\n[DTS] Final results:")
    print(f"  Best f      = {dts_res.best_y:.6e}")
    print(f"  ||x_best||  = {np.linalg.norm(dts_res.best_x):.6e}")
    print(f"  True evals  = {dts_res.evals}")
    print(f"  Final alpha = {dts_res.alpha_final:.3f}")

    print("\n=== Plain CMA-ES baseline (same popsize_factor) ===")
    cma_baseline = DtsCmaEs(
        f=objective,
        x0=x0,
        sigma0=sigma0,
        dim=dim,
        alpha0=1.0,          # irrelevant, plain CMA-ES is used here
        adaptive_alpha=False,
        popsize_factor=2.0,
        verbose=True,
    )
    # Force plain CMA-ES by pretending sklearn is missing:
    baseline_res = cma_baseline._optimize_plain_cma(
        max_evals=max_evals, seed=seed, disp_gap_evals=2000
    )

    print("\n[CMA] Final results:")
    print(f"  Best f      = {baseline_res.best_y:.6e}")
    print(f"  ||x_best||  = {np.linalg.norm(baseline_res.best_x):.6e}")
    print(f"  True evals  = {baseline_res.evals}")

    # Optional: simple textual comparison
    print("\n=== Summary ===")
    print(
        f"DTS-CMA-ES: best f = {dts_res.best_y:.3e} "
        f"after {dts_res.evals} true evaluations"
    )
    print(
        f"Plain CMA-ES: best f = {baseline_res.best_y:.3e} "
        f"after {baseline_res.evals} true evaluations"
    )

    # If you want to plot, uncomment the following:
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.semilogy(dts_res.history_evals, dts_res.history_best, label="DTS-CMA-ES")
    #     plt.semilogy(
    #         baseline_res.history_evals, baseline_res.history_best, label="CMA-ES"
    #     )
    #     plt.xlabel("True function evaluations")
    #     plt.ylabel("Best f(x)")
    #     plt.grid(True, which="both")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    # except ImportError:
    #     print("matplotlib not available; skipping plots.")


def main():
    run_experiment_100d()


if __name__ == "__main__":
    main()
