from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .lshade import LSHADEOptimizer
from .surrogate import MimicSurrogateRanker


class MSLSHADEOptimizer(LSHADEOptimizer):
    """Mimic Surrogate L-SHADE: one true evaluation per generation chosen by a noisy surrogate ranker."""

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        pop_size: int = 100,
        memory_size: Optional[int] = None,
        p_max: float = 0.2,
        p_min: Optional[float] = None,
        min_pop_size: Optional[int] = None,
        top_k: int = 5,
        top_pick: int = 2,
        swap_prob: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dim=dim,
            seed=seed,
            pop_size=pop_size,
            memory_size=memory_size,
            p_max=p_max,
            p_min=p_min,
            min_pop_size=min_pop_size,
            **kwargs,
        )
        self.surrogate = MimicSurrogateRanker(top_k=top_k, top_pick=top_pick, swap_prob=swap_prob, rng=self.rng)

    def tell_one(
        self,
        trial: np.ndarray,
        trial_fit: float,
        info: Tuple[int, float, float],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """
        Update state with a single evaluated trial.
        Parameter adaptation uses parent comparison; replacement uses worst elimination.
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

        # Worst replacement policy
        worst_idx = int(np.argmax(st.fitness))
        worst_fit = float(st.fitness[worst_idx])
        if trial_fit < worst_fit:
            st.archive = np.vstack((st.archive, st.population[worst_idx][None, :]))
            st.population[worst_idx] = trial
            st.fitness[worst_idx] = trial_fit
            if trial_fit < st.best_f:
                st.best_f = trial_fit
                st.best_x = trial.copy()

        # Trim archive to current population size
        if st.archive.shape[0] > st.population.shape[0]:
            remove_count = st.archive.shape[0] - st.population.shape[0]
            idx_remove = rng.choice(st.archive.shape[0], size=remove_count, replace=False)
            st.archive = np.delete(st.archive, idx_remove, axis=0)

        # Memory update
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
        """Run MS-LSHADE until max_evals is reached (one true eval per generation)."""
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
            # Surrogate ranking (cheating evaluation) does not count toward eval budget.
            pred_order, _ = self.surrogate.rank(problem, trials, eval_batch=eval_batch)
            best_idx = int(pred_order[0])
            best_trial = trials[best_idx]
            best_info = info[best_idx]

            best_trial_clipped = np.clip(best_trial, lower, upper)
            best_trial_fit = float(eval_batch([best_trial_clipped])[0])
            st.evals += 1

            self.tell_one(best_trial_clipped, best_trial_fit, best_info, lower, upper)
            # Apply linear population size reduction after each generation
            self._reduce_population(max_evals)

            history.append(st.best_f)
            if st.evals >= max_evals:
                break

        return (
            self._truncate_solution(np.asarray(st.best_x, dtype=float)),
            float(st.best_f),
            history,
        )
