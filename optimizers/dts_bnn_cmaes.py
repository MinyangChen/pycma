from __future__ import annotations

"""DTS-CMA-ES with a Bayesian Neural Network (BNN) surrogate.

This file intentionally **does not** modify/overwrite the existing
``DTSCMAESOptimizer`` implementation. Instead, it introduces a drop-in
replacement optimizer that:

- keeps the *outer DTS flow* identical (warmup -> train model1 -> predict ->
  select n_orig true evals -> add to archive -> train model2 -> predict -> mix
  true+pred -> prediction_guard -> tell -> optional alpha adaptation)
- replaces the sklearn Gaussian Process surrogate with a Bayesian Neural Network
  surrogate implemented using ``torchbnn``.

The new optimizer is suitable for high-dimensional runs (e.g., Ackley 100D) and
can train on GPU (A100) with optional AMP.
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

# Ensure repo root on sys.path when run as a script.
if __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Torch / torchbnn are new dependencies for this optimizer.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR

    import torchbnn as bnn

    _TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    ExponentialLR = None  # type: ignore
    bnn = None  # type: ignore

    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc

# We reuse the DTS-CMA-ES outer loop unchanged via inheritance.
try:
    from .dts_cmaes import DTSCMAESOptimizer
except Exception:  # pragma: no cover
    from dts_cmaes import DTSCMAESOptimizer  # type: ignore


# =============================================================================
# BNN surrogate
# =============================================================================


def _resolve_device(device: str) -> "torch.device":
    """Resolve device string into a torch.device."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "This optimizer requires 'torch' and 'torchbnn'. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )

    dev = str(device).lower().strip()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # user-specified
    if dev.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(dev)


def _build_bnn_model(
    dim_in: int,
    hidden_sizes: Tuple[int, ...],
    prior_sigma: float,
) -> "nn.Module":
    """Build a small Bayes-by-backprop MLP using torchbnn BayesLinear."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "This optimizer requires 'torch' and 'torchbnn'. "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )

    layers: list[nn.Module] = []
    in_f = int(dim_in)
    for h in hidden_sizes:
        h_i = int(h)
        layers.append(bnn.BayesLinear(prior_mu=0.0, prior_sigma=float(prior_sigma), in_features=in_f, out_features=h_i))
        layers.append(nn.Tanh())
        in_f = h_i

    layers.append(bnn.BayesLinear(prior_mu=0.0, prior_sigma=float(prior_sigma), in_features=in_f, out_features=1))
    return nn.Sequential(*layers)


class _BNNSurrogate:
    """Bayesian Neural Network surrogate.

    - Expects whitened inputs Xw (as produced by DTS whitening).
    - Standardizes y internally (y -> (y - mean) / std) and returns predictions on
      the original y-scale.
    - Uses Monte-Carlo forward passes to estimate predictive mean and variance.

    Notes
    -----
    *This is designed as a drop-in surrogate for DTSSurrogateRanker:*
    ``predict()`` returns (mean, variance).
    """

    def __init__(
        self,
        dim: int,
        *,
        prior_sigma: float = 0.1,
        hidden_sizes: Optional[Tuple[int, ...]] = None,
        mc_samples: int = 30,
        epochs: int = 400,
        lr: float = 5e-2,
        lr_gamma: float = 0.999,
        kl_weight: float = 0.1,
        weight_decay: float = 0.0,
        y_std_min: float = 1e-12,
        device: str = "auto",
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0,
        early_stop: bool = True,
        early_stop_min_epochs: int = 200,
        early_stop_win_smooth: int = 200,
        early_stop_win_div: int = 200,
        early_stop_thresh: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "DTSBNN_CMAESOptimizer requires 'torch' and 'torchbnn'. "
                f"Original import error: {_TORCH_IMPORT_ERROR}"
            )

        self.dim = int(dim)

        # Model/training hyperparams
        self.prior_sigma = float(prior_sigma)
        if hidden_sizes is None:
            # Template-inspired default: 2D, 2D
            hidden_sizes = (2 * self.dim, 2 * self.dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        self.mc_samples = int(mc_samples)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.lr_gamma = float(lr_gamma)
        self.kl_weight = float(kl_weight)
        self.weight_decay = float(weight_decay)
        self.y_std_min = float(y_std_min)

        self.device = _resolve_device(device)
        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        self.grad_clip = float(grad_clip) if grad_clip is not None else None

        self.early_stop = bool(early_stop)
        self.early_stop_min_epochs = int(early_stop_min_epochs)
        self.early_stop_win_smooth = int(early_stop_win_smooth)
        self.early_stop_win_div = int(early_stop_win_div)
        self.early_stop_thresh = float(early_stop_thresh)

        self.seed = seed

        # Learned state
        self.model: Optional[nn.Module] = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.is_trained_: bool = False

        # A100-friendly: allow TF32 if available
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                # torch>=2.0
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def fit(self, Xw: np.ndarray, y: np.ndarray) -> bool:
        """Fit BNN on (Xw, y) where Xw is whitened space."""
        Xw = np.asarray(Xw, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if Xw.ndim != 2 or Xw.shape[1] != self.dim:
            raise ValueError(f"Xw must have shape (n, {self.dim}). Got {Xw.shape}.")
        if Xw.shape[0] != y.shape[0] or Xw.shape[0] < 2:
            return False

        # Standardize targets
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        y_std = float(np.std(y_centered))
        if not np.isfinite(y_std) or y_std < self.y_std_min:
            return False

        self.y_mean_ = y_mean
        self.y_std_ = y_std
        y_norm = (y_centered / y_std).astype(np.float32)

        # Determinism (optional)
        if self.seed is not None:
            try:
                torch.manual_seed(int(self.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.seed))
            except Exception:
                pass

        # Build a fresh model each fit (as requested: fit-from-scratch is acceptable)
        self.model = _build_bnn_model(self.dim, self.hidden_sizes, self.prior_sigma).to(self.device)

        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma)

        x_t = torch.from_numpy(Xw).to(self.device)
        y_t = torch.from_numpy(y_norm).to(self.device).view(-1, 1)

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        mse_hist: list[float] = []
        self.model.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(x_t)
                mse = mse_loss(pred, y_t)
                kl = kl_loss(self.model)
                loss = mse + (self.kl_weight * kl)

            scaler.scale(loss).backward()

            if self.grad_clip is not None:
                try:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                except Exception:
                    pass

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            mse_val = float(mse.detach().cpu().item())
            mse_hist.append(mse_val)

            # Early stopping rule adapted from the provided template.
            if self.early_stop:
                if epoch >= self.early_stop_min_epochs:
                    w_smo = self.early_stop_win_smooth
                    w_div = self.early_stop_win_div
                    if len(mse_hist) >= (w_smo + w_div + 2):
                        curve = np.asarray(mse_hist, dtype=np.float64)
                        # Reference window (older)
                        ref = curve[-(w_smo + w_div + 1) : -(w_div + 1)].mean()
                        # Current window (recent)
                        cur = curve[-(w_smo + 1) : -1].mean()
                        if ref > 0.0 and (cur / ref) >= (1.0 - self.early_stop_thresh):
                            break

        self.is_trained_ = True
        return True

    def predict(self, Xw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict (mean, variance) at Xw (whitened space)."""
        if not self.is_trained_ or self.model is None:
            raise RuntimeError("BNN not trained")

        Xw = np.asarray(Xw, dtype=np.float32)
        if Xw.ndim == 1:
            Xw = Xw[None, :]
        if Xw.ndim != 2 or Xw.shape[1] != self.dim:
            raise ValueError(f"Xw must have shape (n, {self.dim}). Got {Xw.shape}.")

        x_t = torch.from_numpy(Xw).to(self.device)

        # We want MC sampling from the posterior. For torchbnn layers, sampling is
        # typically enabled in train() mode. We keep gradients off.
        self.model.train()

        with torch.no_grad():
            # Collect samples: shape (S, N)
            S = max(1, int(self.mc_samples))
            preds = []
            for _ in range(S):
                out = self.model(x_t).view(-1)
                preds.append(out)
            samples = torch.stack(preds, dim=0)  # (S, N)

            mu_norm = torch.mean(samples, dim=0)
            var_norm = torch.var(samples, dim=0, unbiased=False)

        mu = mu_norm.detach().cpu().numpy().astype(np.float64)
        var = var_norm.detach().cpu().numpy().astype(np.float64)

        # De-normalize to original y-scale
        mu = mu * self.y_std_ + self.y_mean_
        var = var * (self.y_std_ ** 2)

        # Numerical safety
        var = np.maximum(var, 1e-18)
        return mu.reshape(-1), var.reshape(-1)


# =============================================================================
# Model bundle compatible with DTSCMAESOptimizer.optimize()
# =============================================================================


@dataclass
class _BNNModelBundle:
    """A bundle holding the surrogate plus whitening state.

    This mirrors the attributes used by DTSCMAESOptimizer.optimize():
    - ``gp``: must expose ``predict(Xw) -> (mean, var)``
    - whitening state: mean/sigma/C
    - y_train_min/y_train_max: used by DTSSurrogateRanker for cpoi/cei
    - age: used by cache aging

    The attribute is named ``gp`` on purpose so that the parent optimize() can
    remain unchanged.
    """

    gp: _BNNSurrogate
    mean: np.ndarray
    sigma: float
    C: np.ndarray
    y_train_min: float
    y_train_max: float
    age: int = 0


# =============================================================================
# DTSBNN_CMAESOptimizer
# =============================================================================


class DTSBNN_CMAESOptimizer(DTSCMAESOptimizer):
    """Drop-in DTS-CMA-ES variant using a BNN surrogate instead of a GP."""

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        # ---- BNN hyperparams ----
        bnn_prior_sigma: float = 0.1,
        bnn_hidden_sizes: Optional[Tuple[int, ...]] = None,
        bnn_mc_samples: int = 30,
        bnn_epochs: int = 400,
        bnn_lr: float = 5e-2,
        bnn_lr_gamma: float = 0.999,
        bnn_kl_weight: float = 0.1,
        bnn_weight_decay: float = 0.0,
        bnn_y_std_min: float = 1e-12,
        bnn_device: str = "auto",
        bnn_use_amp: bool = True,
        bnn_grad_clip: Optional[float] = 1.0,
        bnn_early_stop: bool = True,
        bnn_early_stop_min_epochs: int = 200,
        bnn_early_stop_win_smooth: int = 200,
        bnn_early_stop_win_div: int = 200,
        bnn_early_stop_thresh: float = 0.02,
        # ---- DTS training set size (override defaults for high-D) ----
        n_max_train: Optional[int] = 1000,
        warmup_min_points: Optional[int] = 500,
        # ---- forward everything else to DTSCMAESOptimizer ----
        **kwargs: Any,
    ) -> None:
        # Note: we keep DTSCMAESOptimizer fully intact and only override the
        # surrogate training routine. We still call super().__init__ so that the
        # DTS loop, ranker, whitening, alpha adaptation, etc. remain identical.
        super().__init__(
            dim=dim,
            seed=seed,
            n_max_train=n_max_train,
            warmup_min_points=warmup_min_points,
            **kwargs,
        )

        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "DTSBNN_CMAESOptimizer requires 'torch' and 'torchbnn'. "
                f"Original import error: {_TORCH_IMPORT_ERROR}"
            )

        # Store BNN hyperparams
        self.bnn_prior_sigma = float(bnn_prior_sigma)
        self.bnn_hidden_sizes = tuple(bnn_hidden_sizes) if bnn_hidden_sizes is not None else None
        self.bnn_mc_samples = int(bnn_mc_samples)
        self.bnn_epochs = int(bnn_epochs)
        self.bnn_lr = float(bnn_lr)
        self.bnn_lr_gamma = float(bnn_lr_gamma)
        self.bnn_kl_weight = float(bnn_kl_weight)
        self.bnn_weight_decay = float(bnn_weight_decay)
        self.bnn_y_std_min = float(bnn_y_std_min)
        self.bnn_device = str(bnn_device)
        self.bnn_use_amp = bool(bnn_use_amp)
        self.bnn_grad_clip = float(bnn_grad_clip) if bnn_grad_clip is not None else None

        self.bnn_early_stop = bool(bnn_early_stop)
        self.bnn_early_stop_min_epochs = int(bnn_early_stop_min_epochs)
        self.bnn_early_stop_win_smooth = int(bnn_early_stop_win_smooth)
        self.bnn_early_stop_win_div = int(bnn_early_stop_win_div)
        self.bnn_early_stop_thresh = float(bnn_early_stop_thresh)

    # ------------------------------------------------------------------
    # Override only the surrogate training method.
    # ------------------------------------------------------------------

    def _train_gp_model(  # type: ignore[override]
        self,
        X_archive: np.ndarray,
        y_archive: np.ndarray,
        mean: np.ndarray,
        sigma: float,
        C: np.ndarray,
    ) -> Optional[_BNNModelBundle]:
        """Train a BNN surrogate on the locally-selected archive subset.

        This method intentionally has the same signature as
        ``DTSCMAESOptimizer._train_gp_model`` so that the parent optimize() loop
        can be reused unchanged.
        """

        if X_archive.shape[0] < self.n_min_train:
            return None

        # Whitening and local selection logic are inherited from DTSCMAESOptimizer.
        X_white = self._whiten(X_archive, mean, sigma, C)
        r_A_max = self._compute_r_A_max()
        idx = self._select_training_indices(X_white, r_A_max)
        if idx.size < self.n_min_train:
            return None

        X_train = X_white[idx]
        y_train = y_archive[idx]

        surrogate = _BNNSurrogate(
            dim=self.dim,
            prior_sigma=self.bnn_prior_sigma,
            hidden_sizes=self.bnn_hidden_sizes,
            mc_samples=self.bnn_mc_samples,
            epochs=self.bnn_epochs,
            lr=self.bnn_lr,
            lr_gamma=self.bnn_lr_gamma,
            kl_weight=self.bnn_kl_weight,
            weight_decay=self.bnn_weight_decay,
            y_std_min=self.bnn_y_std_min,
            device=self.bnn_device,
            use_amp=self.bnn_use_amp,
            grad_clip=self.bnn_grad_clip,
            early_stop=self.bnn_early_stop,
            early_stop_min_epochs=self.bnn_early_stop_min_epochs,
            early_stop_win_smooth=self.bnn_early_stop_win_smooth,
            early_stop_win_div=self.bnn_early_stop_win_div,
            early_stop_thresh=self.bnn_early_stop_thresh,
            seed=self.seed,
        )

        ok = surrogate.fit(X_train, y_train)
        if not ok:
            return None

        return _BNNModelBundle(
            gp=surrogate,
            mean=np.asarray(mean, dtype=float).copy(),
            sigma=float(sigma),
            C=np.asarray(C, dtype=float).copy(),
            y_train_min=float(np.min(y_train)),
            y_train_max=float(np.max(y_train)),
            age=0,
        )


__all__ = ["DTSBNN_CMAESOptimizer"]
