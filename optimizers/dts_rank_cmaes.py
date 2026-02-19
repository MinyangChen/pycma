# optimizers/dts_rank_cmaes.py
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

# --- New surrogate deps ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore
    _torch_import_error = exc

try:
    # Package import style (recommended)
    from .base import BaseOptimizer
    from .surrogate import DTSSurrogateRanker
except Exception:  # pragma: no cover
    # Script-mode fallback (allows running this file directly)
    from base import BaseOptimizer  # type: ignore
    from surrogate import DTSSurrogateRanker  # type: ignore


# =============================================================================
# Numeric helpers (no scipy)
# =============================================================================

def _norm_ppf(p: float) -> float:
    """Acklam inverse normal CDF approximation."""
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
    """Approximate Chi-square quantile via Wilson–Hilferty transform."""
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
    y2 assumed more accurate. Uses 0-based ranks.
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
    """Bajer et al. (2019) regression models Q2_min, Q3_max for epsilon bounds."""
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
# Rank diagnostics helpers
# =============================================================================

def _rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    """Rankdata with average ranks for ties. No scipy dependency."""
    a = np.asarray(a, dtype=float).ravel()
    n = a.size
    if n == 0:
        return a.copy()

    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)

    # average ties
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j)
            ranks[order[i:j + 1]] = avg
        i = j + 1

    return ranks


def _spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation computed via Pearson correlation on ranks."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")

    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


# =============================================================================
# Small internal structures (archive, model bundle)
# =============================================================================

@dataclass
class _ModelBundle:
    model: "_RankEnsembleSurrogate"
    mean: np.ndarray
    sigma: float
    C: np.ndarray
    y_train_min: float
    y_train_max: float
    val_spearman: float = float("nan")
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


# =============================================================================
# Ranking-focused surrogate: Deep Ensemble RankNet (+ optional ListNet)
# =============================================================================

class _RankNetMLP(nn.Module):
    def __init__(self, dim: int, hidden_sizes: Tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = int(dim)
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.LayerNorm(int(h)))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _RankEnsembleSurrogate:
    """
    Ranking-focused surrogate producing (mean, var) via a deep ensemble.

    - Inputs: whitened space (Mahalanobis-whitened using CMA mean/sigma/C).
    - Targets: standardized y (mean/std computed on training subset).
    - Training objective: RankNet pairwise logistic + optional ListNet listwise + small Huber regression.
    - Uncertainty: ensemble variance + residual noise floor.

    This is tuned to maximize ranking accuracy rather than RMSE.
    """

    def __init__(
        self,
        dim: int,
        ensemble_size: int = 5,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        dropout: float = 0.10,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 60,
        batch_size: int = 256,
        pairs_per_batch: int = 2048,
        focus_top_frac: float = 0.30,
        listwise_tau: float = 0.75,
        w_pair: float = 1.0,
        w_list: float = 0.5,
        w_reg: float = 0.10,
        val_fraction: float = 0.20,
        patience: int = 10,
        device: str = "auto",  # "auto" / "cpu" / "cuda"
        use_amp: bool = True,
        bootstrap: bool = True,
        y_std_min: float = 1e-12,
        var_floor: float = 1e-12,
        random_state: Optional[int] = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError(f"PyTorch is required for DTS_RANK_CMAESOptimizer: {_torch_import_error}")

        self.dim = int(dim)
        self.ensemble_size = int(max(1, ensemble_size))
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.pairs_per_batch = int(pairs_per_batch)
        self.focus_top_frac = float(focus_top_frac)
        self.listwise_tau = float(listwise_tau)
        self.w_pair = float(w_pair)
        self.w_list = float(w_list)
        self.w_reg = float(w_reg)
        self.val_fraction = float(val_fraction)
        self.patience = int(patience)
        self.use_amp = bool(use_amp)
        self.bootstrap = bool(bootstrap)
        self.y_std_min = float(y_std_min)
        self.var_floor = float(var_floor)
        self.random_state = random_state

        self.device = self._resolve_device(device)
        self.models: List[_RankNetMLP] = []
        self.is_trained_: bool = False

        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.noise_var_y_: float = 0.0
        self.val_spearman_: float = float("nan")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        dev = str(device).lower().strip()
        if dev == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev in ("cuda", "cuda:0"):
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _pairwise_ranknet_loss(
        pred: torch.Tensor,
        y: torch.Tensor,
        pairs_per_batch: int,
        focus_top_frac: float,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """
        RankNet pairwise logistic loss:
            t = sign(y_j - y_i)
            s = pred_j - pred_i
            loss = softplus(-t*s) = log(1 + exp(-t*s))
        Focus sampling on top fraction (lowest y) to improve top-ranking accuracy.
        """
        b = int(y.numel())
        if b < 2 or pairs_per_batch <= 0:
            return torch.zeros((), device=y.device)

        k = max(1, int(round(float(focus_top_frac) * b)))
        if k >= b:
            k = b

        # Select anchors mostly from top-k (best) according to y (lower is better)
        if 0.0 < focus_top_frac < 1.0 and k < b:
            top_idx = torch.topk(-y, k=k, largest=True).indices
            ii = top_idx[torch.randint(0, k, (pairs_per_batch,), generator=gen, device=y.device)]
        else:
            ii = torch.randint(0, b, (pairs_per_batch,), generator=gen, device=y.device)

        jj = torch.randint(0, b, (pairs_per_batch,), generator=gen, device=y.device)

        # avoid i == j
        same = (ii == jj)
        if torch.any(same):
            jj[same] = (jj[same] + 1) % b

        dy = y[jj] - y[ii]
        t = torch.sign(dy)
        mask = t != 0
        if not torch.any(mask):
            return torch.zeros((), device=y.device)

        s = pred[jj] - pred[ii]
        return F.softplus(-t[mask] * s[mask]).mean()

    @staticmethod
    def _listwise_listnet_loss(pred: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
        """
        ListNet-style listwise loss using softmax distributions.
        For minimization, use -y as relevance.
        """
        b = int(y.numel())
        if b < 2:
            return torch.zeros((), device=y.device)
        tau = float(max(1e-6, tau))
        p_true = torch.softmax(-y / tau, dim=0)                 # target distribution
        log_p_pred = torch.log_softmax(-pred / tau, dim=0)      # predicted distribution
        return -(p_true * log_p_pred).sum()

    def fit(self, Xw: np.ndarray, y: np.ndarray) -> Tuple[bool, float]:
        """
        Fit the ensemble. Returns (ok, val_spearman_best).
        """
        Xw = np.asarray(Xw, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        n = int(Xw.shape[0])
        if n < 8 or Xw.shape[1] != self.dim:
            self.is_trained_ = False
            return False, float("nan")

        # Standardize target for stable optimization (ranking is scale-invariant; regression needs stable scale)
        y_mean = float(np.mean(y))
        y_centered = y - y_mean
        y_std = float(np.std(y_centered))
        if y_std < self.y_std_min:
            self.is_trained_ = False
            return False, float("nan")

        self.y_mean_ = y_mean
        self.y_std_ = y_std
        y_norm = (y_centered / y_std).astype(np.float32, copy=False)

        rng = np.random.default_rng(self.random_state)
        idx = np.arange(n, dtype=int)
        rng.shuffle(idx)
        n_val = int(max(2, round(self.val_fraction * n)))
        n_val = min(n_val, n - 2)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        X_tr = torch.as_tensor(Xw[tr_idx], device=self.device)
        y_tr = torch.as_tensor(y_norm[tr_idx], device=self.device)
        X_val = torch.as_tensor(Xw[val_idx], device=self.device)
        y_val = torch.as_tensor(y_norm[val_idx], device=self.device)

        # AMP scaler (if CUDA)
        use_amp = bool(self.use_amp and self.device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.models = []
        best_spearmans: List[float] = []
        val_mses: List[float] = []

        for m in range(self.ensemble_size):
            # Reproducible but diverse seeds
            torch_seed = int(rng.integers(1, 2**31 - 1))
            torch.manual_seed(torch_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(torch_seed)

            model = _RankNetMLP(dim=self.dim, hidden_sizes=self.hidden_sizes, dropout=self.dropout).to(self.device)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            # bootstrap sample for diversity
            if self.bootstrap:
                bs_idx = rng.integers(0, X_tr.shape[0], size=X_tr.shape[0], endpoint=False)
                X_tr_i = X_tr[bs_idx]
                y_tr_i = y_tr[bs_idx]
            else:
                X_tr_i = X_tr
                y_tr_i = y_tr

            best_state: Optional[Dict[str, torch.Tensor]] = None
            best_rho = -float("inf")
            bad = 0

            # Torch generator for pair sampling (device-aware)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(torch_seed)

            for epoch in range(self.epochs):
                model.train()

                n_train_i = int(X_tr_i.shape[0])
                bs = min(self.batch_size, n_train_i)
                perm = torch.randperm(n_train_i, generator=gen, device=self.device)

                for start in range(0, n_train_i, bs):
                    bidx = perm[start:start + bs]
                    xb = X_tr_i[bidx]
                    yb = y_tr_i[bidx]

                    opt.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred = model(xb)

                        # Ranking losses
                        loss_pair = self._pairwise_ranknet_loss(
                            pred=pred, y=yb,
                            pairs_per_batch=self.pairs_per_batch,
                            focus_top_frac=self.focus_top_frac,
                            gen=gen,
                        )
                        loss_list = self._listwise_listnet_loss(pred=pred, y=yb, tau=self.listwise_tau)

                        # Light regression stabilizer (Huber)
                        loss_reg = F.smooth_l1_loss(pred, yb)

                        loss = self.w_pair * loss_pair + self.w_list * loss_list + self.w_reg * loss_reg

                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

                # ---- validation Spearman early stopping ----
                model.eval()
                with torch.no_grad():
                    pred_val = model(X_val).detach().float().cpu().numpy()
                    y_val_np = y_val.detach().float().cpu().numpy()
                    rho = _spearmanr_np(pred_val, y_val_np)

                if not np.isfinite(rho):
                    rho = -1.0

                if rho > best_rho + 1e-6:
                    best_rho = float(rho)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

            # restore best epoch
            if best_state is not None:
                model.load_state_dict(best_state)

            # final val MSE for noise floor estimate
            model.eval()
            with torch.no_grad():
                pred_val = model(X_val).detach()
                mse_val = torch.mean((pred_val - y_val) ** 2).item()

            self.models.append(model)
            best_spearmans.append(best_rho)
            val_mses.append(float(mse_val))

        # Ensemble diagnostics
        self.val_spearman_ = float(np.mean(best_spearmans)) if best_spearmans else float("nan")
        mse_bar = float(np.mean(val_mses)) if val_mses else 0.0
        self.noise_var_y_ = max(0.0, mse_bar * (self.y_std_ ** 2))

        self.is_trained_ = True
        return True, self.val_spearman_

    def predict(self, Xw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (mean_y, var_y) on original y scale.
        """
        if not self.is_trained_ or not self.models:
            raise RuntimeError("Rank surrogate not trained")

        Xw = np.asarray(Xw, dtype=np.float32)
        X = torch.as_tensor(Xw, device=self.device)

        preds: List[np.ndarray] = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                p = model(X).detach().float().cpu().numpy()
            # denormalize
            p_y = p * self.y_std_ + self.y_mean_
            preds.append(p_y)

        P = np.vstack([p.reshape(1, -1) for p in preds])  # (M, N)
        mean = np.mean(P, axis=0)
        var = np.var(P, axis=0) + float(self.noise_var_y_)  # add noise floor
        var = np.maximum(var, self.var_floor)
        return mean.astype(float), var.astype(float)


# =============================================================================
# DTS_RANK_CMAESOptimizer (outer flow identical to DTSCMAESOptimizer)
# =============================================================================

class DTS_RANK_CMAESOptimizer(BaseOptimizer):
    """
    DTS-CMA-ES with ranking-focused deep ensemble surrogate.

    Outer flow is preserved:
      warmup → train model1 → predict pop → select n_orig true eval
      → add to archive → train model2 → predict pop → mix true+pred
      → prediction_guard → tell → optional alpha adaptation.

    Budget counts ONLY true evaluations.
    """

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        # ---- CMA-ES init ----
        sigma0: float = 3.0,
        x0: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        pop_size: Optional[int] = None,
        popsize_mode: str = "default",  # "default" / "double"
        cma_options: Optional[Dict[str, Any]] = None,
        # ---- DTS knobs ----
        alpha0: float = 0.5,
        use_adaptive_alpha: bool = False,
        beta: float = 0.3,
        alpha_min: float = 0.04,
        alpha_max: float = 1.0,
        min_true_per_gen: int = 1,
        warmup_min_points: Optional[int] = 1000,
        n_min_train: Optional[int] = None,         # default: max(5, 3D)
        n_max_train: Optional[int] = 1200,         # allow bigger local sets for ranking
        radius_mode: str = "chi2",                 # "chi2" or "sqrt_dim"
        radius_chi2_p: float = 0.99,
        r_A_max_factor: float = 4.0,
        selection_criterion: str = "cstd",         # "cstd"/"mean"/"cpoi"/"cei"
        cpoi_eps: float = 0.05,
        prediction_guard: str = "shift",           # "shift"/"clamp"/"none"
        use_model_cache: bool = True,
        max_model_age: int = 2,
        whiten_jitter: float = 1e-12,
        require_min_points_in_radius: bool = False,
        # ---- NEW: training subset composition ----
        global_train_frac: float = 0.10,           # small fraction of global points
        global_best_k: int = 50,                   # also include top-k best archive points if not in local
        # ---- NEW: rank surrogate hyperparams ----
        rank_ensemble_size: int = 5,
        rank_hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        rank_dropout: float = 0.10,
        rank_lr: float = 3e-4,
        rank_weight_decay: float = 1e-4,
        rank_epochs: int = 60,
        rank_batch_size: int = 256,
        rank_pairs_per_batch: int = 2048,
        rank_focus_top_frac: float = 0.30,
        rank_listwise_tau: float = 0.75,
        rank_w_pair: float = 1.0,
        rank_w_list: float = 0.5,
        rank_w_reg: float = 0.10,
        rank_val_fraction: float = 0.20,
        rank_patience: int = 10,
        rank_device: str = "auto",
        rank_use_amp: bool = True,
        rank_bootstrap: bool = True,
        rank_y_std_min: float = 1e-12,
        rank_var_floor: float = 1e-12,
        # ---- Logging ----
        print_every: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=dim, seed=seed)

        if torch is None:  # pragma: no cover
            raise ImportError(f"PyTorch is required for DTS_RANK_CMAESOptimizer: {_torch_import_error}")

        # CMA / init
        self.sigma0 = float(sigma0)
        self.x0 = np.zeros(dim, dtype=float) if x0 is None else np.asarray(x0, dtype=float).reshape(dim)
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

        # Dimension-dependent defaults
        n_min = int(n_min_train or max(5, 3 * dim))
        n_max = int(n_max_train or min(20 * dim, 1200))
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

        self.require_min_points_in_radius = bool(require_min_points_in_radius)

        # Training subset strategy
        self.global_train_frac = float(global_train_frac)
        self.global_best_k = int(global_best_k)

        # Rank surrogate params
        self.rank_ensemble_size = int(rank_ensemble_size)
        self.rank_hidden_sizes = tuple(int(h) for h in rank_hidden_sizes)
        self.rank_dropout = float(rank_dropout)
        self.rank_lr = float(rank_lr)
        self.rank_weight_decay = float(rank_weight_decay)
        self.rank_epochs = int(rank_epochs)
        self.rank_batch_size = int(rank_batch_size)
        self.rank_pairs_per_batch = int(rank_pairs_per_batch)
        self.rank_focus_top_frac = float(rank_focus_top_frac)
        self.rank_listwise_tau = float(rank_listwise_tau)
        self.rank_w_pair = float(rank_w_pair)
        self.rank_w_list = float(rank_w_list)
        self.rank_w_reg = float(rank_w_reg)
        self.rank_val_fraction = float(rank_val_fraction)
        self.rank_patience = int(rank_patience)
        self.rank_device = str(rank_device)
        self.rank_use_amp = bool(rank_use_amp)
        self.rank_bootstrap = bool(rank_bootstrap)
        self.rank_y_std_min = float(rank_y_std_min)
        self.rank_var_floor = float(rank_var_floor)

        # Ranker (decoupled in surrogate.py)
        self.ranker = DTSSurrogateRanker(mode=self.selection_criterion, cpoi_eps=self.cpoi_eps)

        # Printing
        self.print_every = int(print_every)

        # Outputs after optimize()
        self.stop_reasons_: Dict[str, Any] = {}
        self.evals_: int = 0
        self.history_: List[float] = []

        # Diagnostics
        self.spearman_orig_: List[float] = []
        self.rde_mu_hist_: List[float] = []
        self.val_spearman_model1_: List[float] = []
        self.val_spearman_model2_: List[float] = []

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

    def _compose_training_indices(
        self,
        idx_local: np.ndarray,
        y_archive: np.ndarray,
        n_total: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Add a small fraction of global points (random) + a few best points (lowest y)
        to improve stability / calibration without losing locality.
        """
        idx_local = np.asarray(idx_local, dtype=int)
        if idx_local.size == 0:
            return idx_local

        # Start with local
        chosen = set(int(i) for i in idx_local.tolist())

        # Add best-k global points (by y)
        if self.global_best_k > 0 and y_archive.size == n_total:
            k = min(self.global_best_k, n_total)
            best_idx = np.argsort(y_archive)[:k]
            for i in best_idx:
                chosen.add(int(i))

        # Add random global fraction
        if self.global_train_frac > 0.0:
            target_extra = int(round(self.global_train_frac * float(idx_local.size)))
            target_extra = max(0, target_extra)
            # Don't exceed overall cap
            cap = int(self.n_max_train)
            max_extra = max(0, cap - len(chosen))
            target_extra = min(target_extra, max_extra)

            if target_extra > 0:
                pool = np.array([i for i in range(n_total) if i not in chosen], dtype=int)
                if pool.size > 0:
                    take = min(target_extra, pool.size)
                    extra = rng.choice(pool, size=take, replace=False)
                    for i in extra:
                        chosen.add(int(i))

        idx = np.array(sorted(chosen), dtype=int)
        # If we exceeded n_max_train due to best_k, trim by "locality": keep closest local points + best points
        if idx.size > self.n_max_train:
            # Keep local points first, then fill remaining with best points
            idx = idx[: self.n_max_train]
        return idx.astype(int)

    def _train_rank_model(
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
        idx_local = self._select_training_indices(X_white, r_A_max)
        if idx_local.size < self.n_min_train:
            return None

        rng = np.random.default_rng(self.seed)
        idx = self._compose_training_indices(idx_local, y_archive, X_archive.shape[0], rng=rng)

        X_train = X_white[idx]
        y_train = y_archive[idx]

        model = _RankEnsembleSurrogate(
            dim=self.dim,
            ensemble_size=self.rank_ensemble_size,
            hidden_sizes=self.rank_hidden_sizes,
            dropout=self.rank_dropout,
            lr=self.rank_lr,
            weight_decay=self.rank_weight_decay,
            epochs=self.rank_epochs,
            batch_size=self.rank_batch_size,
            pairs_per_batch=self.rank_pairs_per_batch,
            focus_top_frac=self.rank_focus_top_frac,
            listwise_tau=self.rank_listwise_tau,
            w_pair=self.rank_w_pair,
            w_list=self.rank_w_list,
            w_reg=self.rank_w_reg,
            val_fraction=self.rank_val_fraction,
            patience=self.rank_patience,
            device=self.rank_device,
            use_amp=self.rank_use_amp,
            bootstrap=self.rank_bootstrap,
            y_std_min=self.rank_y_std_min,
            var_floor=self.rank_var_floor,
            random_state=self.seed,
        )

        ok, val_rho = model.fit(X_train, y_train)
        if not ok:
            return None

        return _ModelBundle(
            model=model,
            mean=np.asarray(mean, dtype=float).copy(),
            sigma=float(sigma),
            C=np.asarray(C, dtype=float).copy(),
            y_train_min=float(np.min(y_train)),
            y_train_max=float(np.max(y_train)),
            val_spearman=float(val_rho),
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
        if cma is None:  # pragma: no cover
            raise ImportError(f"pycma (package 'cma') is required: {_cma_import_error}")

        if max_evals <= 0:
            return self._truncate_solution(np.zeros(self.dim)), float("inf"), []

        eval_batch = self._build_batch_evaluator(problem)

        x0 = self.x0.copy()
        sigma0 = float(self.sigma0)

        # CMA-ES options
        opts = dict(self.cma_options)
        opts.setdefault("popsize", self._resolve_popsize())
        opts.setdefault("seed", self.seed)
        opts.setdefault("verb_log", 0)
        opts.setdefault("verb_disp", 0)

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

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

        while not es.stop() and eval_count < max_evals:
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
                    break
                y_true = np.asarray(eval_batch(list(X_pop)), dtype=float).reshape(-1)
                archive.add_batch(X_pop, y_true)
                eval_count += lamb
                es.tell(X_pop_list, list(y_true))

                _, best_f = archive.best()
                history.append(best_f)
                if self.print_every > 0 and (gen % self.print_every == 0):
                    print(f"[DTS-RANK] iter {gen:5d} | true_evals {eval_count:7d} | best f = {best_f:.6e} | warmup")
                continue

            # Optional: require min points inside radius
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
                        print(f"[DTS-RANK] iter {gen:5d} | true_evals {eval_count:7d} | best f = {best_f:.6e} | density_gate")
                    continue

            # Train model1 (or fallback to cache)
            X_arch, y_arch = archive.as_arrays()
            bundle1 = self._train_rank_model(X_arch, y_arch, mean, sigma, C)
            bundle1_is_new = bundle1 is not None
            if bundle1 is None and self.use_model_cache and model_cache is not None:
                bundle1 = model_cache
                bundle1_is_new = False

            if bundle1 is None:
                # Fallback: full eval generation
                if remaining < lamb:
                    break
                y_true = np.asarray(eval_batch(list(X_pop)), dtype=float).reshape(-1)
                archive.add_batch(X_pop, y_true)
                eval_count += lamb
                es.tell(X_pop_list, list(y_true))

                _, best_f = archive.best()
                history.append(best_f)
                if self.print_every > 0 and (gen % self.print_every == 0):
                    print(f"[DTS-RANK] iter {gen:5d} | true_evals {eval_count:7d} | best f = {best_f:.6e} | no_model")
                continue

            self.val_spearman_model1_.append(float(bundle1.val_spearman))

            # model1 predict
            Xw1 = self._whiten(X_pop, bundle1.mean, bundle1.sigma, bundle1.C)
            mu1, var1 = bundle1.model.predict(Xw1)
            std1 = np.sqrt(np.maximum(var1, 1e-18))

            # Decide number of true evaluations this generation
            n_orig_target = int(math.ceil(alpha * lamb))
            n_orig_target = max(self.min_true_per_gen, n_orig_target)
            n_orig_target = min(n_orig_target, lamb)

            n_orig = min(n_orig_target, remaining)
            if n_orig <= 0:
                break

            # Select points (existing ranker interface preserved)
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

            # Diagnostics: Spearman on truly evaluated points (out-of-sample wrt model1)
            rho_orig = _spearmanr_np(mu1[idx_orig], y_orig)
            self.spearman_orig_.append(float(rho_orig))

            # Train model2 on updated archive (fallback to model1 if fail)
            X_arch2, y_arch2 = archive.as_arrays()
            bundle2 = self._train_rank_model(X_arch2, y_arch2, mean, sigma, C)
            bundle2_is_new = bundle2 is not None
            skip_adapt = False
            if bundle2 is None:
                bundle2 = bundle1
                bundle2_is_new = False
                skip_adapt = True

            self.val_spearman_model2_.append(float(bundle2.val_spearman))

            # model2 predict & mix
            Xw2 = self._whiten(X_pop, bundle2.mean, bundle2.sigma, bundle2.C)
            mu2, var2 = bundle2.model.predict(Xw2)

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
            mu_eff = getattr(es.sp, "mu", max(1, lamb // 2))
            eps_rde = _rde_mu(mu1, y_mix, mu=int(mu_eff))
            self.rde_mu_hist_.append(float(eps_rde))

            if self.use_adaptive_alpha and not skip_adapt:
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
                print(
                    f"[DTS-RANK] iter {gen:5d} | true_evals {eval_count:7d} | best f = {best_f:.6e} "
                    f"| alpha={alpha:.3f} | spearman(orig)={rho_orig:.3f} | RDE_mu={eps_rde:.3f} "
                    f"| val_rho(m1)={bundle1.val_spearman:.3f} | val_rho(m2)={bundle2.val_spearman:.3f}"
                )

        best_x, best_f = archive.best()
        if best_x is None:
            best_x = np.zeros(self.dim, dtype=float)
            best_f = float("inf")

        self.stop_reasons_ = dict(es.stop())
        self.evals_ = int(eval_count)
        self.history_ = list(history)

        return self._truncate_solution(best_x), float(best_f), history
