from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import BaseOptimizer


@dataclass
class _ShadeState:
    population: np.ndarray
    fitness: np.ndarray
    archive: np.ndarray
    memory_F: np.ndarray
    memory_CR: np.ndarray
    memory_pos: int
    best_x: np.ndarray
    best_f: float
    evals: int
    generation: int = 0


class SHADEOptimizer(BaseOptimizer):
    """Success-History Adapted Differential Evolution (SHADE) wrapper."""

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        pop_size: int = 100,
        memory_size: Optional[int] = None,
        p_max: float = 0.2,
        p_min: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=dim, seed=seed)
        self.pop_size = pop_size
        self.memory_size = memory_size or pop_size
        self.p_max = p_max
        self.p_min = p_min if p_min is not None else 2.0 / pop_size
        self.state: Optional[_ShadeState] = None

    @staticmethod
    def _sample_cauchy(loc: float, scale: float, rng: np.random.Generator) -> float:
        """Sample F ~ Cauchy(loc, scale). Resample if <= 0; clamp to 1 if > 1."""
        for _ in range(100):
            val = loc + scale * rng.standard_cauchy()
            if val > 0:
                return 1.0 if val > 1.0 else val
        return 1e-3

    def _init_state(self, problem: Any) -> _ShadeState:
        rng = self.rng
        dim = self.dim

        lower_bound = getattr(problem, "lower_bound", None)
        upper_bound = getattr(problem, "upper_bound", None)
        if lower_bound is None or upper_bound is None:
            # Fallback bounds for unbounded problems.
            lower = np.full(dim, -1.0)
            upper = np.full(dim, 1.0)
        else:
            lower = np.asarray(lower_bound, dtype=float)
            upper = np.asarray(upper_bound, dtype=float)
            if lower.size == 1:
                lower = np.full(dim, float(lower))
            if upper.size == 1:
                upper = np.full(dim, float(upper))

        bounds_range = upper - lower
        population = lower + bounds_range * rng.random((self.pop_size, dim))

        eval_batch = self._build_batch_evaluator(problem)
        fitness = np.asarray(eval_batch(population), dtype=float)
        evals = len(fitness)

        best_idx = int(np.argmin(fitness))
        best_val = float(fitness[best_idx])
        best_vec = population[best_idx].copy()

        memory_F = np.full(self.memory_size, 0.5)
        memory_CR = np.full(self.memory_size, 0.5)

        archive = np.empty((0, dim))

        return _ShadeState(
            population=population,
            fitness=fitness,
            archive=archive,
            memory_F=memory_F,
            memory_CR=memory_CR,
            memory_pos=0,
            best_x=best_vec,
            best_f=best_val,
            evals=evals,
            generation=0,
        )

    def ask(self) -> Tuple[List[np.ndarray], List[Tuple[int, float, float]]]:
        if self.state is None:
            raise RuntimeError("Optimizer state not initialized. Call optimize first.")

        rng = self.rng
        st = self.state
        dim = self.dim
        trials: List[np.ndarray] = []
        info: List[Tuple[int, float, float]] = []

        order = np.argsort(st.fitness)

        for i in range(self.pop_size):
            m_idx = rng.integers(0, self.memory_size)
            Fi = self._sample_cauchy(st.memory_F[m_idx], 0.1, rng)

            if st.memory_CR[m_idx] < 0:
                CRi = 0.0
            else:
                CRi = float(np.clip(rng.normal(st.memory_CR[m_idx], 0.1), 0.0, 1.0))

            p_i = rng.uniform(self.p_min, self.p_max)
            p_num = max(2, int(np.floor(p_i * self.pop_size)))
            pbest_idx = int(order[rng.integers(0, p_num)])

            cand_r1 = np.arange(self.pop_size)
            cand_r1 = cand_r1[(cand_r1 != i) & (cand_r1 != pbest_idx)]
            r1_idx = int(rng.choice(cand_r1))

            pool_size = self.pop_size + st.archive.shape[0]
            pool_indices = np.arange(pool_size)

            forbid = {i, r1_idx, pbest_idx}
            mask = np.ones(pool_size, dtype=bool)
            for idx in forbid:
                if idx < pool_size:
                    mask[idx] = False
            pool_indices = pool_indices[mask]

            if pool_indices.size == 0:
                alt = np.arange(self.pop_size)
                alt = alt[(alt != i) & (alt != r1_idx)]
                r2_flat = int(rng.choice(alt))
            else:
                r2_flat = int(rng.choice(pool_indices))

            if r2_flat < self.pop_size:
                x_r2 = st.population[r2_flat]
            else:
                x_r2 = st.archive[r2_flat - self.pop_size]

            x_i = st.population[i]
            x_pbest = st.population[pbest_idx]
            x_r1 = st.population[r1_idx]

            mutant = x_i + Fi * (x_pbest - x_i) + Fi * (x_r1 - x_r2)

            j_rand = rng.integers(0, dim)
            cross_mask = rng.random(dim) <= CRi
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, mutant, x_i)

            # bounds will be handled in tell using stored bounds
            trials.append(trial)
            info.append((i, Fi, CRi))

        return trials, info

    def tell(self, trials: List[np.ndarray], trial_fitness: List[float], info: List[Tuple[int, float, float]], lower: np.ndarray, upper: np.ndarray) -> None:
        if self.state is None:
            raise RuntimeError("Optimizer state not initialized. Call optimize first.")
        st = self.state
        rng = self.rng

        success_F: List[float] = []
        success_CR: List[float] = []
        success_delta: List[float] = []

        for trial, fit, (i, Fi, CRi) in zip(trials, trial_fitness, info):
            trial = np.clip(trial, lower, upper)
            trial_fit = float(fit)

            if trial_fit <= st.fitness[i]:
                prev_fit = float(st.fitness[i])

                st.archive = np.vstack((st.archive, st.population[i][None, :]))

                st.population[i] = trial
                st.fitness[i] = trial_fit

                success_F.append(Fi)
                success_CR.append(CRi)
                success_delta.append(prev_fit - trial_fit)

                if trial_fit < st.best_f:
                    st.best_f = trial_fit
                    st.best_x = trial.copy()

        if st.archive.shape[0] > self.pop_size:
            remove_count = st.archive.shape[0] - self.pop_size
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

            num_F = float(np.sum(w * (sF**2)))
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

        history: List[float] = [st.best_f]

        eval_batch = self._build_batch_evaluator(problem)

        while st.evals < max_evals:
            trials, info = self.ask()
            # Clip before evaluation so vectors and fitness stay consistent.
            clipped_trials = [np.clip(t, lower, upper) for t in trials]
            trial_fitness = eval_batch(clipped_trials)
            st.evals += len(trial_fitness)
            self.tell(clipped_trials, trial_fitness, info, lower, upper)
            history.append(st.best_f)
            if st.evals >= max_evals:
                break

        return self._truncate_solution(np.asarray(st.best_x, dtype=float)), float(st.best_f), history
