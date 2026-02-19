from __future__ import annotations

"""DTS-BRQ-CMA-ES: DTS-CMA-ES with a lightweight Bayesian Ridge quadratic surrogate.

This optimizer is intended as a drop-in alternative to :class:`DTSCMAESOptimizer` for
high-dimensional settings (e.g., 100D) where a full Gaussian Process surrogate can be
too slow / memory heavy.

Surrogate model:
    - Work in CMA whitened space (same transform as DTS-CMA-ES).
    - Fit a *diagonal quadratic* regression using features [1, z, z^2].
    - Use sklearn.linear_model.BayesianRidge to obtain predictive mean and uncertainty.

The outer DTS-CMA-ES flow is unchanged:
    warmup -> train model1 -> rank/select n_orig -> true evals -> train model2 ->
    predict/mix -> prediction guard -> tell -> optional alpha adaptation.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import BayesianRidge

try:
    # Package import style (recommended)
    from .dts_cmaes import DTSCMAESOptimizer
except Exception:  # pragma: no cover
    # Script-mode fallback
    from dts_cmaes import DTSCMAESOptimizer  # type: ignore


class _BayesianRidgeQuadraticSurrogate:
    """Lightweight local surrogate: diagonal quadratic regression + BayesianRidge.

    Notes
    -----
    * Expects whitened CMA coordinates as input (z = Sigma^{-1/2}(x-mean)).
    * Standardizes y before fitting to stabilize hyperparameter estimation.
    * Predict returns (mean, var) on the original y scale.
    """

    def __init__(
        self,
        dim: int,
        y_std_min: float = 1e-12,
        fit_intercept: bool = False,
    ) -> None:
        self.dim = int(dim)
        self.y_std_min = float(y_std_min)

        # We explicitly include a constant feature in the design matrix; therefore
        # keep fit_intercept=False by default to avoid double intercept.
        self.model = BayesianRidge(fit_intercept=bool(fit_intercept))

        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.is_trained_: bool = False

        self._iu = np.triu_indices(self.dim)  # (i,j) for i<=j, cache it

    # def _features(self, Z: np.ndarray) -> np.ndarray:
    #     """Build diagonal-quadratic features Phi = [1, z, z^2]."""
    #     Z = np.asarray(Z, dtype=float)
    #     if Z.ndim == 1:
    #         Z = Z[None, :]
    #     if Z.shape[1] != self.dim:
    #         raise ValueError(f"Expected Z.shape[1] == {self.dim}, got {Z.shape[1]}")

    #     ones = np.ones((Z.shape[0], 1), dtype=float)
    #     Z2 = Z * Z
    #     return np.hstack([ones, Z, Z2])

    def _features(self, Z: np.ndarray) -> np.ndarray:
        """Build full quadratic features Phi = [1, z, vec(z z^T upper-tri)]."""
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        if Z.shape[1] != self.dim:
            raise ValueError(f"Expected Z.shape[1] == {self.dim}, got {Z.shape[1]}")

        ones = np.ones((Z.shape[0], 1), dtype=float)

        # upper-tri quadratic terms (includes squares and cross terms)
        i, j = self._iu
        Z_quad = Z[:, i] * Z[:, j]  # shape: (n, D*(D+1)/2)

        return np.hstack([ones, Z, Z_quad])

    def fit(self, Z: np.ndarray, y: np.ndarray) -> bool:
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if Z.shape[0] != y.shape[0] or Z.shape[0] < 2:
            return False

        Phi = self._features(Z)

        # Standardize y (like the GP wrapper) for stability.
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
                self.model.fit(Phi, y_norm)
            self.is_trained_ = True
            return True
        except Exception:
            self.is_trained_ = False
            return False

    def predict(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, var) on original y scale."""
        if not self.is_trained_:
            raise RuntimeError("BayesianRidge surrogate not trained")

        Phi = self._features(Z)
        mean_norm, std_norm = self.model.predict(Phi, return_std=True)

        mean = mean_norm * self.y_std_ + self.y_mean_
        var = (std_norm * self.y_std_) ** 2
        return mean, var


@dataclass
class _BRQModelBundle:
    """Bundle with the same attributes expected by DTSCMAESOptimizer.optimize()."""

    gp: _BayesianRidgeQuadraticSurrogate  # keep attribute name `gp` for compatibility
    mean: np.ndarray
    sigma: float
    C: np.ndarray
    y_train_min: float
    y_train_max: float
    age: int = 0


class DTSBRQ_CMAESOptimizer(DTSCMAESOptimizer):
    """DTS-CMA-ES variant using BayesianRidge Quadratic surrogate (BRQ)."""

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        # ---- BRQ surrogate knobs ----
        br_candidate_pool: int = 2500,
        br_y_std_min: float = 1e-12,
        br_fit_intercept: bool = False,
        **kwargs: Any,
    ) -> None:
        # Initialize DTS-CMA-ES core (CMA params, DTS params, etc.)
        super().__init__(dim=dim, seed=seed, **kwargs)

        # Surrogate-specific settings
        self.br_candidate_pool = int(br_candidate_pool)
        self.br_y_std_min = float(br_y_std_min)
        self.br_fit_intercept = bool(br_fit_intercept)

    # ------------------------------------------------------------------
    # Override only the surrogate training method; keep the DTS flow intact.
    # ------------------------------------------------------------------

    def _train_gp_model(
        self,
        X_archive: np.ndarray,
        y_archive: np.ndarray,
        mean: np.ndarray,
        sigma: float,
        C: np.ndarray,
    ) -> Optional[_BRQModelBundle]:
        """Train the BRQ surrogate on a *local* subset of the archive.

        We keep the same method name as the parent class ("_train_gp_model") so the
        inherited optimize() method and all DTS logic remain unchanged.
        """

        if X_archive.shape[0] < self.n_min_train:
            return None

        # Optional speed-up: only consider the most recent points as a candidate pool
        # before applying the Mahalanobis/whitened distance selection.
        if self.br_candidate_pool > 0 and X_archive.shape[0] > self.br_candidate_pool:
            X_pool = X_archive[-self.br_candidate_pool :]
            y_pool = y_archive[-self.br_candidate_pool :]
        else:
            X_pool = X_archive
            y_pool = y_archive

        # Whiten using current CMA state (same as DTS-CMA-ES)
        X_white = self._whiten(X_pool, mean, sigma, C)

        # Local subset selection using inherited distance logic
        r_A_max = self._compute_r_A_max()
        idx = self._select_training_indices(X_white, r_A_max)
        if idx.size < self.n_min_train:
            return None

        Z_train = X_white[idx]
        y_train = y_pool[idx]

        model = _BayesianRidgeQuadraticSurrogate(
            dim=self.dim,
            y_std_min=self.br_y_std_min,
            fit_intercept=self.br_fit_intercept,
        )
        ok = model.fit(Z_train, y_train)
        if not ok:
            return None

        return _BRQModelBundle(
            gp=model,
            mean=np.asarray(mean, dtype=float).copy(),
            sigma=float(sigma),
            C=np.asarray(C, dtype=float).copy(),
            y_train_min=float(np.min(y_train)),
            y_train_max=float(np.max(y_train)),
            age=0,
        )
