from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .base import BaseOptimizer

from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class _GaussianProcessSurrogate:
    """Compact GP wrapper with input/output standardization."""

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
            raise RuntimeError("Input transform is not initialized.")
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
            raise RuntimeError("GP is not trained.")
        X = np.asarray(X, dtype=float)
        Xn = self._transform_x(X)
        mu_n, std_n = self.gp.predict(Xn, return_std=True)
        mu = mu_n * self.y_std_ + self.y_mean_
        std = np.maximum(std_n, 1e-12) * self.y_std_
        return mu, std


@dataclass
class _DEGPState:
    population: np.ndarray
    fitness: np.ndarray
    train_X: np.ndarray
    train_y: np.ndarray
    best_x: np.ndarray
    best_f: float
    evals: int
    generation: int = 0


class DEGPOptimizer(BaseOptimizer):
    """Classic DE with GP ranking and top-K true evaluations per generation."""

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        pop_size: Optional[int] = None,
        alpha_size: Optional[int] = None,
        train_size: Optional[int] = None,
        F: float = 0.5,
        CR: float = 0.9,
        k_true: int = 5,
        strategy: str = "rand",  # "rand", "best", "current-to-best"
        rank_mode: str = "lcb",  # "lcb" or "mean"
        kappa: float = 1.0,
        gp_nu: float = 2.5,
        gp_n_restarts_optimizer: int = 0,
        gp_random_state: Optional[int] = None,
        gp_y_std_min: float = 1e-12,
        print_every: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=dim, seed=seed)

        base_size = max(1, 5 * int(dim))
        if pop_size is None:
            pop_size = base_size
        if alpha_size is None:
            alpha_size = base_size
        if train_size is None:
            train_size = base_size

        self.pop_size = int(pop_size)
        self.alpha_size = int(alpha_size)  # initial true archive size
        self.train_size = int(train_size)  # tau: training archive cap

        if self.pop_size < 4:
            raise ValueError("DEGPOptimizer requires pop_size >= 4.")
        if self.pop_size > self.alpha_size:
            raise ValueError("pop_size must be <= alpha_size to initialize from the alpha archive.")

        self.F = float(F)
        if self.F <= 0:
            raise ValueError("F must be > 0.")

        self.CR = float(CR)
        if self.CR < 0.0 or self.CR > 1.0:
            raise ValueError("CR must be in [0, 1].")

        self.k_true = max(1, int(k_true))

        self.strategy = str(strategy).lower()
        if self.strategy not in {"rand", "best", "current-to-best"}:
            raise ValueError("strategy must be one of: 'rand', 'best', 'current-to-best'.")

        self.rank_mode = str(rank_mode).lower()
        if self.rank_mode not in {"lcb", "mean"}:
            raise ValueError("rank_mode must be 'lcb' or 'mean'.")

        self.kappa = float(kappa)
        self.gp_nu = float(gp_nu)
        self.gp_n_restarts_optimizer = int(gp_n_restarts_optimizer)
        self.gp_random_state = gp_random_state
        self.gp_y_std_min = float(gp_y_std_min)
        self.print_every = int(print_every)

        self.state: Optional[_DEGPState] = None

    def _get_bounds(self, problem: Any) -> Tuple[np.ndarray, np.ndarray]:
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
        return lower, upper

    def _init_state(self, problem: Any, max_evals: int) -> Tuple[_DEGPState, List[float]]:
        eval_batch = self._build_batch_evaluator(problem)
        lower, upper = self._get_bounds(problem)

        n_init = min(self.alpha_size, int(max_evals))
        # Latin Hypercube Sampling in [0, 1]^D, then scale to [lower, upper].
        jitter = self.rng.random((n_init, self.dim))
        perms = np.column_stack([self.rng.permutation(n_init) for _ in range(self.dim)])
        lhs_unit = (perms + jitter) / float(max(n_init, 1))
        X_init = lower + (upper - lower) * lhs_unit
        y_init = np.asarray(eval_batch(X_init), dtype=float).reshape(-1)

        n_pop = min(self.pop_size, n_init)
        if n_pop <= 0:
            raise RuntimeError("No initialization evaluations available.")

        order = np.argsort(y_init)
        pop_idx = order[:n_pop]
        population = X_init[pop_idx].copy()
        fitness = y_init[pop_idx].copy()

        best_idx = int(np.argmin(fitness))
        best_x = population[best_idx].copy()
        best_f = float(fitness[best_idx])

        history: List[float] = []
        running_best = float("inf")
        for yi in y_init:
            if yi < running_best:
                running_best = float(yi)
            history.append(running_best)

        state = _DEGPState(
            population=population,
            fitness=fitness,
            train_X=X_init.copy(),
            train_y=y_init.copy(),
            best_x=best_x,
            best_f=best_f,
            evals=n_init,
            generation=0,
        )
        return state, history

    def _ask_trials(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Optimizer state not initialized.")

        st = self.state
        pop = st.population
        fit = st.fitness
        n_pop = pop.shape[0]
        dim = self.dim
        rng = self.rng

        best_idx = int(np.argmin(fit))
        x_best = pop[best_idx]

        trials = np.empty_like(pop)

        for i in range(n_pop):
            if self.strategy == "rand":
                candidates = np.arange(n_pop)
                candidates = candidates[candidates != i]
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

            elif self.strategy == "best":
                candidates = np.arange(n_pop)
                mask = np.ones(n_pop, dtype=bool)
                mask[i] = False
                mask[best_idx] = False
                candidates = candidates[mask]
                if candidates.size < 2:
                    candidates = np.arange(n_pop)
                    candidates = candidates[candidates != i]
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                mutant = x_best + self.F * (pop[r1] - pop[r2])

            else:  # "current-to-best"
                x_i = pop[i]
                candidates = np.arange(n_pop)
                mask = np.ones(n_pop, dtype=bool)
                mask[i] = False
                if best_idx != i:
                    mask[best_idx] = False
                candidates = candidates[mask]
                if candidates.size < 2:
                    candidates = np.arange(n_pop)
                    candidates = candidates[candidates != i]
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                mutant = x_i + self.F * (x_best - x_i) + self.F * (pop[r1] - pop[r2])

            x_i = pop[i]
            j_rand = rng.integers(0, dim)
            cross_mask = rng.random(dim) <= self.CR
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, mutant, x_i)
            trials[i] = trial

        return trials

    def _rank_trials(self, trials: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Optimizer state not initialized.")
        st = self.state
        n_trials = int(trials.shape[0])

        gp = _GaussianProcessSurrogate(
            dim=self.dim,
            nu=self.gp_nu,
            n_restarts_optimizer=self.gp_n_restarts_optimizer,
            random_state=self.gp_random_state,
            y_std_min=self.gp_y_std_min,
        )
        ok = gp.fit(st.train_X, st.train_y)
        if not ok:
            return self.rng.permutation(n_trials)

        mu, std = gp.predict(trials)
        if self.rank_mode == "lcb":
            score = mu - self.kappa * std
        else:
            score = mu
        return np.argsort(score).astype(int)

    def _insert_training_point(self, x: np.ndarray, y: float) -> None:
        if self.state is None:
            raise RuntimeError("Optimizer state not initialized.")
        st = self.state

        st.train_X = np.vstack((st.train_X, np.asarray(x, dtype=float).reshape(1, self.dim)))
        st.train_y = np.append(st.train_y, float(y))

        if st.train_y.size > self.train_size:
            excess = st.train_y.size - self.train_size
            worst_idx = np.argsort(st.train_y)[-excess:]
            keep = np.ones(st.train_y.size, dtype=bool)
            keep[worst_idx] = False
            st.train_X = st.train_X[keep]
            st.train_y = st.train_y[keep]

    def optimize(self, problem: Any, max_evals: int) -> Tuple[np.ndarray, float, List[float]]:
        if max_evals <= 0:
            return self._truncate_solution(np.zeros(self.dim)), float("inf"), []

        self.state, history = self._init_state(problem, max_evals=max_evals)
        st = self.state
        if st is None:
            return self._truncate_solution(np.zeros(self.dim)), float("inf"), []

        if st.evals >= max_evals:
            return self._truncate_solution(st.best_x), float(st.best_f), history
        if st.population.shape[0] < 4:
            return self._truncate_solution(st.best_x), float(st.best_f), history

        eval_batch = self._build_batch_evaluator(problem)
        lower, upper = self._get_bounds(problem)

        while st.evals < max_evals:
            trials = self._ask_trials()
            trials = np.clip(trials, lower, upper)

            pred_order = self._rank_trials(trials)
            remaining = max_evals - st.evals
            k_eval = min(self.k_true, int(pred_order.size), int(remaining))
            if k_eval <= 0:
                break

            selected_idx = np.asarray(pred_order[:k_eval], dtype=int)
            selected_trials = trials[selected_idx]
            selected_fits = np.asarray(eval_batch(selected_trials), dtype=float).reshape(-1)

            for trial, trial_fit in zip(selected_trials, selected_fits):
                trial_fit = float(trial_fit)
                st.evals += 1

                worst_idx = int(np.argmax(st.fitness))
                worst_fit = float(st.fitness[worst_idx])
                if trial_fit < worst_fit:
                    st.population[worst_idx] = trial
                    st.fitness[worst_idx] = trial_fit

                    if trial_fit < st.best_f:
                        st.best_f = trial_fit
                        st.best_x = trial.copy()

                self._insert_training_point(trial, trial_fit)
                history.append(float(st.best_f))

                if st.evals >= max_evals:
                    break

            st.generation += 1
            if self.print_every > 0 and (st.generation % self.print_every == 0):
                print(
                    f"[DE-GP] iter {st.generation:4d} | true_evals {st.evals:6d} | best f = {st.best_f:.6e}"
                )

            if st.evals >= max_evals:
                break

        return self._truncate_solution(st.best_x), float(st.best_f), history
