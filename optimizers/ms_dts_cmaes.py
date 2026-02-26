from __future__ import annotations

"""
MS-DTS-CMA-ES (Mimic-Surrogate DTS-CMA-ES)

This optimizer keeps the same control flow, warmup logic, evaluation accounting,
printing cadence, seeding, CMA options, stop bookkeeping, and history tracking
as DTSCMAESOptimizer, but replaces GP training/prediction with a mimic surrogate
ranker (MimicSurrogateRanker) that "cheats" by evaluating the true objective and
then slightly scrambling the ranking.

IMPORTANT:
- Cheating evaluations inside MimicSurrogateRanker MUST NOT count toward eval budget
  and MUST NOT be stored in the archive.
- Only the selected idx_orig points are truly evaluated (counted) and stored.
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root on sys.path when run as a script.
if __package__ is None:  # pragma: no cover
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

try:
    import cma  # pycma
except Exception as exc:  # pragma: no cover
    cma = None  # type: ignore
    _cma_import_error = exc

try:
    # Package import style (recommended)
    from .base import BaseOptimizer
    from .surrogate import MimicSurrogateRanker
except Exception:  # pragma: no cover
    # Script-mode fallback (allows running this file directly)
    from base import BaseOptimizer  # type: ignore
    from surrogate import MimicSurrogateRanker  # type: ignore


# =============================================================================
# Numeric helpers (copied from DTS; no scipy required)
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
# Archive (same behavior as DTS): stores ONLY true evaluated points
# =============================================================================

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
# MS-DTS-CMA-ES Optimizer
# =============================================================================

class MSDTSCMAESOptimizer(BaseOptimizer):
    """
    MS-DTS-CMA-ES: Mimic-surrogate variant of DTS-CMA-ES (no GP training).

    Keeps the same warmup logic, evaluation counting, printing cadence, seeding,
    CMA options, stop bookkeeping, and history as DTSCMAESOptimizer.

    Differences (only after warmup):
      - Uses MimicSurrogateRanker to obtain a slightly scrambled ranking for the
        whole CMA population (cheating evaluation, NOT counted, NOT archived).
      - Truly evaluates only the top-n_orig candidates (counted + archived).
      - Uses the mimic ranking to build y_mix for es.tell.
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
        # ---- DTS knobs (kept consistent) ----
        alpha0: float = 0.05,
        use_adaptive_alpha: bool = False,
        beta: float = 0.3,
        alpha_min: float = 0.04,
        alpha_max: float = 1.0,
        min_true_per_gen: int = 1,
        warmup_min_points: Optional[int] = 1000,  # default: max(10D, 50)
        n_min_train: Optional[int] = None,        # used by optional radius gate
        n_max_train: Optional[int] = None,        # kept for parity; not otherwise used
        radius_mode: str = "chi2",                # "chi2" or "sqrt_dim"
        radius_chi2_p: float = 0.99,
        r_A_max_factor: float = 4.0,
        whiten_jitter: float = 1e-12,
        require_min_points_in_radius: bool = False,
        # ---- Mimic knobs (ONLY new public knobs) ----
        mimic_top_k: int = 5,
        mimic_top_pick: int = 2,
        mimic_swap_prob: float = 0.2,
        inject_true_into_tell: bool = False,
        # ---- Logging ----
        print_every: int = 100,
        **kwargs: Any,
    ) -> None:
        # Avoid forwarding unknown kwargs to BaseOptimizer
        super().__init__(dim=dim, seed=seed)

        # ---- CMA init ----
        self.sigma0 = float(sigma0)
        if x0 is None:
            self.x0 = np.zeros(dim, dtype=float)  # consistent with your current DTS default
        else:
            self.x0 = np.asarray(x0, dtype=float).reshape(dim)

        self.bounds = bounds
        self.pop_size = pop_size
        self.popsize_mode = str(popsize_mode).lower()

        self.cma_options = dict(cma_options) if cma_options else {}

        # Optional: support legacy runner configs that pass "options" instead of "cma_options".
        # This is minimal-intrusion and helps MS-DTS run in the same runner setups.
        opt_alias = kwargs.get("options", None)
        if opt_alias is not None and isinstance(opt_alias, dict) and not cma_options:
            self.cma_options.update(dict(opt_alias))

        if self.bounds is not None and "bounds" not in self.cma_options:
            self.cma_options["bounds"] = self.bounds

        # ---- DTS knobs ----
        self.alpha0 = float(alpha0)
        self.use_adaptive_alpha = bool(use_adaptive_alpha)
        self.beta = float(beta)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.min_true_per_gen = int(min_true_per_gen)

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
        self.whiten_jitter = float(whiten_jitter)
        self.require_min_points_in_radius = bool(require_min_points_in_radius)

        # ---- Mimic surrogate ranker ----
        self.mimic_top_k = int(mimic_top_k)
        self.mimic_top_pick = int(mimic_top_pick)
        self.mimic_swap_prob = float(mimic_swap_prob)
        self.inject_true_into_tell = bool(inject_true_into_tell)

        self.mimic_ranker = MimicSurrogateRanker(
            top_k=self.mimic_top_k,
            top_pick=self.mimic_top_pick,
            swap_prob=self.mimic_swap_prob,
            rng=self.rng,
        )

        # ---- Logging ----
        self.print_every = int(print_every)

        # ---- Outputs after optimize() ----
        self.stop_reasons_: Dict[str, Any] = {}
        self.evals_: int = 0
        self.history_: List[float] = []

    # ------------------------------------------------------------------ #
    # Internal helpers (kept consistent with DTS)
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
        """
        Whiten points using current CMA covariance:
            z = (sigma^2 C)^{-1/2} (x - mean)
        Used ONLY for the optional local-density gate (require_min_points_in_radius).
        """
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

    # ------------------------------------------------------------------ #
    # Main library API
    # ------------------------------------------------------------------ #

    def optimize(self, problem: Any, max_evals: int) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run MS-DTS-CMA-ES on `problem` with a true-evaluation budget `max_evals`.

        Returns:
            best_x (truncated), best_f, history_best_f
        """
        if cma is None:  # pragma: no cover
            raise ImportError(f"pycma (package 'cma') is required: {_cma_import_error}")

        if max_evals <= 0:
            return self._truncate_solution(np.zeros(self.dim)), float("inf"), []

        eval_batch = self._build_batch_evaluator(problem)

        # Bounds only used to build lower/upper for potential x0 sampling.
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

        # Keep consistent with DTS: default x0 is the origin (set in __init__)
        x0 = self.x0.copy()
        sigma0 = float(self.sigma0)

        # CMA-ES options (same style as DTS)
        opts = dict(self.cma_options)
        opts.setdefault("popsize", self._resolve_popsize())
        opts.setdefault("seed", self.seed)
        opts.setdefault("verb_log", 0)
        opts.setdefault("verb_disp", 0)  # printing handled here

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        archive = _Archive(self.dim)

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
                    break  # can't finish full warmup generation without overshooting budget
                y_true = np.asarray(eval_batch(list(X_pop)), dtype=float).reshape(-1)
                archive.add_batch(X_pop, y_true)
                eval_count += lamb
                es.tell(X_pop_list, list(y_true))

                _, best_f = archive.best()
                history.append(best_f)
                if self.print_every > 0 and (gen % self.print_every == 0):
                    print(f"[MS-DTS] iter {gen:4d} | true_evals {eval_count:6d} | best f = {best_f:.6e}")
                continue

            # Optional local-density gate (kept consistent with DTS)
            if self.require_min_points_in_radius:
                X_arch, _y_arch = archive.as_arrays()
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
                        print(f"[MS-DTS] iter {gen:4d} | true_evals {eval_count:6d} | best f = {best_f:.6e}")
                    continue

            # Decide number of true evaluations this generation (same rule as DTS)
            n_orig_target = int(math.ceil(alpha * lamb))
            n_orig_target = max(self.min_true_per_gen, n_orig_target)
            n_orig_target = min(n_orig_target, lamb)

            n_orig = min(n_orig_target, remaining)
            if n_orig <= 0:
                break

            # (3) Mimic ranking: CHEATING evaluation (must NOT count, must NOT archive)
            pred_order, _true_f_cheat = self.mimic_ranker.rank(
                problem,
                candidates=list(X_pop),
                eval_batch=eval_batch,
            )
            pred_order = np.asarray(pred_order, dtype=int).reshape(-1)
            if pred_order.size != lamb:
                raise RuntimeError("MimicSurrogateRanker returned an invalid ranking size.")

            # (4) Select points for true evaluation
            idx_orig = pred_order[:n_orig]
            X_orig = X_pop[idx_orig]

            # (5) True-evaluate ONLY selected points (counts + archived)
            y_orig = np.asarray(eval_batch(list(X_orig)), dtype=float).reshape(-1)
            archive.add_batch(X_orig, y_orig)
            eval_count += int(y_orig.size)

            # (6) Build mixed fitness vector y_mix based on ranking (no GP)
            y_rank = np.empty(lamb, dtype=float)
            y_rank[pred_order] = np.arange(lamb, dtype=float)  # 0 is best (no ties)
            y_mix = y_rank.copy()

            if self.inject_true_into_tell:
                # Optional: overwrite evaluated points with true fitness values
                y_mix[idx_orig] = y_orig

            # (Optional) adaptive alpha: keep the mechanism, use rank-vs-true mismatch proxy
            # Note: We compute RDE between surrogate ranking (y_rank) and the true objective
            # ranking. We re-compute true_f for the whole population here would violate
            # the budget logic, so we do NOT. Instead, we use the cheating values from the
            # mimic ranker ONLY if it returns them. However, per constraints, cheating values
            # must not be archived or budgeted; using them for alpha adaptation is allowed.
            if self.use_adaptive_alpha:
                # mimic_ranker.rank returned _true_f_cheat (cheating full-pop objective values)
                true_f = np.asarray(_true_f_cheat, dtype=float).reshape(-1)
                if true_f.size == lamb:
                    mu_eff = getattr(es.sp, "mu", max(1, lamb // 2))
                    eps_rde = _rde_mu(y_rank, true_f, mu=int(mu_eff))
                    eps_smooth = (1.0 - self.beta) * eps_smooth + self.beta * eps_rde
                    alpha = _update_alpha_self_adaptive(
                        eps_smooth=eps_smooth,
                        alpha_current=alpha,
                        dim=self.dim,
                        alpha_min=self.alpha_min,
                        alpha_max=self.alpha_max,
                        clamp_dim_to_20=True,
                    )

            # (7) CMA-ES update with rank-based pseudo fitness
            es.tell(X_pop_list, list(y_mix))

            # (8) History + printing (same cadence as DTS)
            _, best_f = archive.best()
            history.append(best_f)
            if self.print_every > 0 and (gen % self.print_every == 0):
                print(f"[MS-DTS] iter {gen:4d} | true_evals {eval_count:6d} | best f = {best_f:.6e}")

        best_x, best_f = archive.best()
        if best_x is None:
            best_x = np.zeros(self.dim, dtype=float)
            best_f = float("inf")

        # Save outputs for inspection (same pattern as DTS)
        self.stop_reasons_ = dict(es.stop())
        self.evals_ = int(eval_count)
        self.history_ = list(history)

        return self._truncate_solution(best_x), float(best_f), history


# =============================================================================
# Demo (library-style): run on AckleyProblem
# =============================================================================

if __name__ == "__main__":  # pragma: no cover
    from problems.ackley import AckleyProblem  # type: ignore

    dim = 20
    max_evals = 1000

    problem = AckleyProblem(dim=dim)

    opt = MSDTSCMAESOptimizer(
        dim=dim,
        seed=0,
        sigma0=9.0,
        bounds=(problem.lower_bound, problem.upper_bound),
        popsize_mode="default",
        alpha0=0.05,
        min_true_per_gen=1,
        warmup_min_points=max(10 * dim, 50),
        print_every=10,
        mimic_top_k=5,
        mimic_top_pick=2,
        mimic_swap_prob=0.2,
        inject_true_into_tell=False,

        cma_options={
            "tolfunhist": 0.0,  # 关闭 funhist 收敛判据
            "tolfun": 0.0,      # 关闭 fun 收敛判据（建议一起关）
            # 可选：避免别的阈值过早触发（通常不用）
            # "tolx": 0.0,
            # "tolstagnation": 1e9,
        },
    )


    best_x, best_f, hist = opt.optimize(problem, max_evals=max_evals)

    print("\n=== MS-DTS-CMA-ES on Ackley ===")
    print(f"best f: {best_f:.6e}")
    print(f"best x (first 10): {best_x}")
    print(f"true evals used: {opt.evals_}")
    print("stop reasons:", opt.stop_reasons_)
