from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .shade import SHADEOptimizer


class LSHADEOptimizer(SHADEOptimizer):
    """L-SHADE: SHADE with Linear Population Size Reduction (Tanabe & Fukunaga)."""

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        pop_size: int = 100,
        memory_size: Optional[int] = None,
        p_max: float = 0.2,
        p_min: Optional[float] = None,
        min_pop_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # L-SHADE needs at least 4 individuals for current-to-pbest/1
        if pop_size < 4:
            raise ValueError("LSHADEOptimizer requires pop_size >= 4.")

        super().__init__(
            dim=dim,
            seed=seed,
            pop_size=pop_size,
            memory_size=memory_size,
            p_max=p_max,
            p_min=p_min,
            **kwargs,
        )

        # Initial and minimal population sizes
        self.init_pop_size = pop_size
        if min_pop_size is None:
            min_pop_size = 4
        # Clamp min_pop_size into [4, pop_size]
        min_pop_size = max(4, min(min_pop_size, pop_size))
        self.min_pop_size = min_pop_size

    def ask(self) -> Tuple[List[np.ndarray], List[Tuple[int, float, float]]]:
        """Generate trial vectors for the current population size."""
        if self.state is None:
            raise RuntimeError("Optimizer state not initialized. Call optimize first.")

        rng = self.rng
        st = self.state
        dim = self.dim
        trials: List[np.ndarray] = []
        info: List[Tuple[int, float, float]] = []

        # Current population size (shrinks over time)
        N = st.population.shape[0]
        order = np.argsort(st.fitness)

        for i in range(N):
            # Sample F and CR from historical memory
            m_idx = rng.integers(0, self.memory_size)
            Fi = self._sample_cauchy(st.memory_F[m_idx], 0.1, rng)

            if st.memory_CR[m_idx] < 0:
                CRi = 0.0
            else:
                CRi = float(np.clip(rng.normal(st.memory_CR[m_idx], 0.1), 0.0, 1.0))

            # p-best selection based on current N
            p_i = rng.uniform(self.p_min, self.p_max)
            p_num = max(2, int(np.floor(p_i * N)))
            p_num = min(p_num, N)
            pbest_idx = int(order[rng.integers(0, p_num)])

            # r1 from population (not i or pbest)
            cand_r1 = np.arange(N)
            cand_r1 = cand_r1[(cand_r1 != i) & (cand_r1 != pbest_idx)]
            r1_idx = int(rng.choice(cand_r1))

            # r2 from population ∪ archive, excluding i, r1, pbest
            pool_size = N + st.archive.shape[0]
            pool_indices = np.arange(pool_size)
            forbid = {i, r1_idx, pbest_idx}
            mask = np.ones(pool_size, dtype=bool)
            for idx in forbid:
                # indices refer to population part only, not archive
                if idx < N:
                    mask[idx] = False
            pool_indices = pool_indices[mask]

            if pool_indices.size == 0:
                # Fallback: choose from population only (not i or r1)
                alt = np.arange(N)
                alt = alt[(alt != i) & (alt != r1_idx)]
                r2_flat = int(rng.choice(alt))
            else:
                r2_flat = int(rng.choice(pool_indices))

            if r2_flat < N:
                x_r2 = st.population[r2_flat]
            else:
                x_r2 = st.archive[r2_flat - N]

            x_i = st.population[i]
            x_pbest = st.population[pbest_idx]
            x_r1 = st.population[r1_idx]

            # current-to-pbest/1 mutation
            mutant = x_i + Fi * (x_pbest - x_i) + Fi * (x_r1 - x_r2)

            # Binomial crossover with at least one mutated dimension
            j_rand = rng.integers(0, dim)
            cross_mask = rng.random(dim) <= CRi
            cross_mask[j_rand] = True
            trial = np.where(cross_mask, mutant, x_i)

            # Bounds will be handled in optimize (before evaluation)
            trials.append(trial)
            info.append((i, Fi, CRi))

        return trials, info

    def _reduce_population(self, max_evals: int) -> None:
        """Apply linear population size reduction."""
        if self.state is None:
            return
        st = self.state

        N_init = self.init_pop_size
        N_min = self.min_pop_size
        current_N = st.population.shape[0]

        if current_N <= N_min or max_evals <= 0:
            return

        # Linear schedule: N(FES) = round(N_init - (N_init - N_min) * FES / MaxFES)
        ratio = min(1.0, st.evals / float(max_evals))
        target_N = int(round(N_init - (N_init - N_min) * ratio))
        target_N = max(N_min, min(target_N, N_init))

        if target_N >= current_N:
            return

        # Keep best target_N individuals
        order = np.argsort(st.fitness)
        keep_idx = order[:target_N]

        st.population = st.population[keep_idx]
        st.fitness = st.fitness[keep_idx]

        # Recompute global best
        best_idx = int(np.argmin(st.fitness))
        st.best_f = float(st.fitness[best_idx])
        st.best_x = st.population[best_idx].copy()

        # Trim archive to at most target_N
        if st.archive.shape[0] > target_N:
            remove_count = st.archive.shape[0] - target_N
            idx_remove = self.rng.choice(st.archive.shape[0], size=remove_count, replace=False)
            st.archive = np.delete(st.archive, idx_remove, axis=0)

    def tell(
        self,
        trials: List[np.ndarray],
        trial_fitness: List[float],
        info: List[Tuple[int, float, float]],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """
        Update population, archive, and parameter memory.

        Assumes `trials` have already been clipped to [lower, upper] in `optimize`.
        """
        if self.state is None:
            raise RuntimeError("Optimizer state not initialized. Call optimize first.")
        st = self.state
        rng = self.rng

        success_F: List[float] = []
        success_CR: List[float] = []
        success_delta: List[float] = []

        # Current population size for this generation
        N = st.population.shape[0]

        for trial, fit, (i, Fi, CRi) in zip(trials, trial_fitness, info):
            # trials are assumed already clipped; do not re-clip to avoid double work
            trial_fit = float(fit)

            if trial_fit < st.fitness[i]:
                # Strict improvement: selection + success logging
                prev_fit = float(st.fitness[i])

                # Add replaced individual to archive
                st.archive = np.vstack((st.archive, st.population[i][None, :]))

                st.population[i] = trial
                st.fitness[i] = trial_fit

                success_F.append(Fi)
                success_CR.append(CRi)
                success_delta.append(prev_fit - trial_fit)

                if trial_fit < st.best_f:
                    st.best_f = trial_fit
                    st.best_x = trial.copy()

            elif trial_fit == st.fitness[i]:
                # Allow equal-fitness replacement but DO NOT log as success
                st.population[i] = trial
                st.fitness[i] = trial_fit

        # Limit archive size to current population size N
        if st.archive.shape[0] > N:
            remove_count = st.archive.shape[0] - N
            idx_remove = rng.choice(st.archive.shape[0], size=remove_count, replace=False)
            st.archive = np.delete(st.archive, idx_remove, axis=0)

        # Historical memory update (same as SHADE, but based on strict improvements)
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
        """Run L-SHADE until max_evals is reached."""
        if max_evals <= 0:
            return self._truncate_solution(np.zeros(self.dim)), float("inf"), []

        self.state = self._init_state(problem)
        st = self.state

        # Bounds
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
            # Clip trials before evaluation so trials and fitness are consistent.
            clipped_trials = [np.clip(t, lower, upper) for t in trials]
            trial_fitness = eval_batch(clipped_trials)
            st.evals += len(trial_fitness)

            self.tell(clipped_trials, trial_fitness, info, lower, upper)
            self._reduce_population(max_evals)

            history.append(st.best_f)
            if st.evals >= max_evals:
                break

        return (
            self._truncate_solution(np.asarray(st.best_x, dtype=float)),
            float(st.best_f),
            history,
        )
