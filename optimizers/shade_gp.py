from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple

import numpy as np

from .shade import SHADEOptimizer

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class _Archive:
    """Store only true objective evaluations collected by the optimizer."""

    def __init__(self, dim: int) -> None:
        self.dim = int(dim)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []

    @property
    def size(self) -> int:
        return len(self.y)

    def add(self, x: np.ndarray, y: float) -> None:
        self.X.append(np.asarray(x, dtype=float).reshape(self.dim).copy())
        self.y.append(float(y))

    def add_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        for xi, yi in zip(X, y):
            self.add(xi, float(yi))

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.X:
            return np.empty((0, self.dim), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(self.X), np.asarray(self.y, dtype=float)


class _GaussianProcessSurrogate:
    """Small GP wrapper with input/output standardization."""

    def __init__(
        self,
        dim: int,
        nu: float,
        n_restarts_optimizer: int,
        random_state: Optional[int],
        y_std_min: float,
    ) -> None:
        self.dim = int(dim)
        self.y_std_min = float(y_std_min)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=nu)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=False,
            n_restarts_optimizer=int(n_restarts_optimizer),
            random_state=random_state,
        )
        self.x_mean_: Optional[np.ndarray] = None
        self.x_scale_: Optional[np.ndarray] = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.is_trained_: bool = False

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_scale_ is None:
            raise RuntimeError("GP input transform is not initialized.")
        return (X - self.x_mean_) / self.x_scale_

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != self.dim:
            return False
        if X.shape[0] != y.size or y.size < 2:
            return False

        self.x_mean_ = np.mean(X, axis=0)
        x_scale = np.std(X, axis=0)
        self.x_scale_ = np.where(x_scale > 1e-12, x_scale, 1.0)
        Xn = self._transform_x(X)

        self.y_mean_ = float(np.mean(y))
        y_centered = y - self.y_mean_
        y_std = float(np.std(y_centered))
        if y_std < self.y_std_min:
            return False
        self.y_std_ = y_std
        yn = y_centered / self.y_std_

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                self.gp.fit(Xn, yn)
            self.is_trained_ = True
            return True
        except Exception:
            self.is_trained_ = False
            return False

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained_:
            raise RuntimeError("GP not trained.")
        X = np.asarray(X, dtype=float)
        Xn = self._transform_x(X)
        mu_n, std_n = self.gp.predict(Xn, return_std=True)
        mu = mu_n * self.y_std_ + self.y_mean_
        std = np.maximum(std_n, 1e-12) * self.y_std_
        return mu, std


class SHADEGPOptimizer(SHADEOptimizer):
    """
    SHADE with a real GP surrogate ranker.

    Per generation:
    1) ask() creates trial candidates
    2) surrogate ranks candidates (no true objective calls in ranking)
    3) only top-ranked trial is truly evaluated
    4) tell_one() updates SHADE state
    """

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        pop_size: int = 100,
        memory_size: Optional[int] = None,
        p_max: float = 0.2,
        p_min: Optional[float] = None,
        warmup_min_points: Optional[int] = None,
        gp_rank_mode: str = "lcb",  # "lcb" or "mean"
        lcb_kappa: float = 1.0,
        gp_max_train_size: int = 300,
        gp_keep_best: int = 60,
        gp_nu: float = 2.5,
        gp_n_restarts_optimizer: int = 0,
        gp_random_state: Optional[int] = None,
        gp_y_std_min: float = 1e-12,
        print_every: int = 300,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dim=dim,
            seed=seed,
            pop_size=pop_size,
            memory_size=memory_size,
            p_max=p_max,
            p_min=p_min,
            **kwargs,
        )
        self.warmup_min_points = int(warmup_min_points or max(5, 3 * dim))
        self.gp_rank_mode = str(gp_rank_mode).lower()
        if self.gp_rank_mode not in {"lcb", "mean"}:
            raise ValueError("gp_rank_mode must be 'lcb' or 'mean'.")
        self.lcb_kappa = float(lcb_kappa)
        self.gp_max_train_size = max(2, int(gp_max_train_size))
        self.gp_keep_best = max(0, int(gp_keep_best))

        self.gp_nu = float(gp_nu)
        self.gp_n_restarts_optimizer = int(gp_n_restarts_optimizer)
        self.gp_random_state = gp_random_state
        self.gp_y_std_min = float(gp_y_std_min)
        self.print_every = int(print_every)

        self.train_archive_: Optional[_Archive] = None

    def _select_training_indices(self, X: np.ndarray, y: np.ndarray, center: np.ndarray) -> np.ndarray:
        n = int(y.size)
        if n <= self.gp_max_train_size:
            return np.arange(n, dtype=int)

        k = self.gp_max_train_size
        keep_best = min(self.gp_keep_best, k, n)

        idx_best = np.argsort(y)[:keep_best] if keep_best > 0 else np.empty((0,), dtype=int)
        remaining = k - idx_best.size
        if remaining <= 0:
            return idx_best.astype(int)

        d2 = np.sum((X - center) ** 2, axis=1)
        near_order = np.argsort(d2)

        if idx_best.size == 0:
            return near_order[:k].astype(int)

        mask = np.ones(n, dtype=bool)
        mask[idx_best] = False
        idx_near = near_order[mask[near_order]][:remaining]
        return np.concatenate((idx_best, idx_near)).astype(int)

    def _rank_trials(self, trials_clipped: np.ndarray, archive: _Archive) -> np.ndarray:
        n_trials = int(trials_clipped.shape[0])
        if archive.size < self.warmup_min_points:
            return self.rng.permutation(n_trials)

        X_all, y_all = archive.as_arrays()
        center = np.mean(trials_clipped, axis=0)
        idx = self._select_training_indices(X_all, y_all, center)
        if idx.size < 2:
            return self.rng.permutation(n_trials)

        gp = _GaussianProcessSurrogate(
            dim=self.dim,
            nu=self.gp_nu,
            n_restarts_optimizer=self.gp_n_restarts_optimizer,
            random_state=self.gp_random_state,
            y_std_min=self.gp_y_std_min,
        )
        ok = gp.fit(X_all[idx], y_all[idx])
        if not ok:
            return self.rng.permutation(n_trials)

        mu, std = gp.predict(trials_clipped)
        if self.gp_rank_mode == "lcb":
            score = mu - self.lcb_kappa * std
        else:
            score = mu
        return np.argsort(score).astype(int)

    def tell_one(
        self,
        trial: np.ndarray,
        trial_fit: float,
        info: Tuple[int, float, float],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """
        Update SHADE state using one true-evaluated trial.
        Parameter adaptation compares with the trial's parent;
        replacement follows worst-elimination policy.
        """
        if self.state is None:
            raise RuntimeError("Optimizer state not initialized. Call optimize first.")
        st = self.state
        rng = self.rng

        i_parent, Fi, CRi = info
        parent_fit = float(st.fitness[i_parent])

        success_F: List[float] = []
        success_CR: List[float] = []
        success_delta: List[float] = []

        if trial_fit < parent_fit:
            success_F.append(Fi)
            success_CR.append(CRi)
            success_delta.append(parent_fit - trial_fit)

        worst_idx = int(np.argmax(st.fitness))
        worst_fit = float(st.fitness[worst_idx])
        if trial_fit < worst_fit:
            st.archive = np.vstack((st.archive, st.population[worst_idx][None, :]))
            st.population[worst_idx] = trial
            st.fitness[worst_idx] = trial_fit
            if trial_fit < st.best_f:
                st.best_f = trial_fit
                st.best_x = trial.copy()

        if st.archive.shape[0] > st.population.shape[0]:
            remove_count = st.archive.shape[0] - st.population.shape[0]
            idx_remove = rng.choice(st.archive.shape[0], size=remove_count, replace=False)
            st.archive = np.delete(st.archive, idx_remove, axis=0)

        if success_F:
            sF = np.asarray(success_F, dtype=float)
            sCR = np.asarray(success_CR, dtype=float)
            dF = np.asarray(success_delta, dtype=float)

            total_impr = float(np.sum(dF))
            if total_impr > 0:
                w = dF / total_impr
            else:
                w = np.full_like(dF, 1.0 / len(dF))

            num_F = float(np.sum(w * (sF ** 2)))
            den_F = float(np.sum(w * sF))
            if den_F > 0:
                st.memory_F[st.memory_pos] = num_F / den_F
            st.memory_CR[st.memory_pos] = float(np.sum(w * sCR))
            st.memory_pos = (st.memory_pos + 1) % self.memory_size

        st.generation += 1

    def optimize(self, problem: Any, max_evals: int) -> Tuple[np.ndarray, float, List[float]]:
        if max_evals <= 0:
            return self._truncate_solution(np.zeros(self.dim)), float("inf"), []

        self.state = self._init_state(problem)
        st = self.state

        lower_bound = getattr(problem, "lower_bound", None)
        upper_bound = getattr(problem, "upper_bound", None)
        if lower_bound is None or upper_bound is None:
            lower = np.full(self.dim, -1.0)
            upper = np.full(self.dim, 1.0)
        else:
            lower = np.asarray(lower_bound, dtype=float)
            upper = np.asarray(upper_bound, dtype=float)
            if lower.size == 1:
                lower = np.full(self.dim, float(lower))
            if upper.size == 1:
                upper = np.full(self.dim, float(upper))

        archive = _Archive(self.dim)
        archive.add_batch(st.population, st.fitness)
        self.train_archive_ = archive

        history: List[float] = [st.best_f]
        eval_batch = self._build_batch_evaluator(problem)

        while st.evals < max_evals:
            trials, info = self.ask()
            trials_clipped = np.clip(np.asarray(trials, dtype=float), lower, upper)

            pred_order = self._rank_trials(trials_clipped, archive)
            best_idx = int(pred_order[0])
            best_trial = trials_clipped[best_idx]
            best_info = info[best_idx]

            best_fit = float(eval_batch([best_trial])[0])
            st.evals += 1

            self.tell_one(best_trial, best_fit, best_info, lower, upper)
            archive.add(best_trial, best_fit)
            history.append(st.best_f)
            if self.print_every > 0 and (st.generation % self.print_every == 0):
                print(
                    f"[SHADE-GP] iter {st.generation:4d} | true_evals {st.evals:6d} | best f = {st.best_f:.6e}"
                )

            if st.evals >= max_evals:
                break

        return (
            self._truncate_solution(np.asarray(st.best_x, dtype=float)),
            float(st.best_f),
            history,
        )


# Optional alias with underscore naming.
SHADE_GPOptimizer = SHADEGPOptimizer
