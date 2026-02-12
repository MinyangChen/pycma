from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


class MimicSurrogateRanker:
    """
    Surrogate ranker that "cheats" by evaluating the true objective and then
    injecting noise into the ranking to mimic an imperfect surrogate.
    """

    def __init__(
        self,
        top_k: int = 5,
        top_pick: int = 2,
        swap_prob: float = 0.2,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.top_k = top_k
        self.top_pick = top_pick
        self.swap_prob = swap_prob
        self.rng = rng or np.random.default_rng()

    def rank(
        self,
        problem: Any,
        candidates: List[np.ndarray],
        eval_batch: Optional[Callable[[List[np.ndarray]], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate candidates on the true objective (cheating) and return a noisy ranking.
        Returns:
            predicted_order: indices (0..N-1) in predicted rank (best first)
            true_fitness: true objective values
        """
        if eval_batch is not None:
            true_f = np.asarray(eval_batch(candidates), dtype=float).reshape(-1)
        else:
            true_f = np.asarray([problem(np.asarray(c)) for c in candidates], dtype=float).reshape(-1)

        N = len(candidates)
        true_order = np.argsort(true_f)

        k = min(self.top_k, N)
        m = min(self.top_pick, k)

        elite_pool = list(true_order[:k])
        chosen_elite: List[int] = []
        if elite_pool and m > 0:
            chosen_elite = list(self.rng.choice(elite_pool, size=m, replace=False))
            if len(chosen_elite) > 1:
                chosen_elite = list(self.rng.permutation(chosen_elite))

        remaining = [idx for idx in true_order if idx not in chosen_elite]
        # Apply local adjacent swaps to mimic noisy ranking
        for i in range(1, len(remaining)):
            if self.rng.random() < self.swap_prob:
                remaining[i - 1], remaining[i] = remaining[i], remaining[i - 1]

        predicted_order = np.array(chosen_elite + remaining, dtype=int)
        return predicted_order, true_f


# =============================================================================
# DTS-CMA-ES: surrogate-based ranker for choosing which points to truly evaluate
# =============================================================================

def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF Φ(z) without scipy."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF φ(z)."""
    z = np.asarray(z, dtype=float)
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z * z)


class DTSSurrogateRanker:
    """
    Rank candidates using GP mean/std (used by DTS-CMA-ES to pick points for true eval).

    Supported modes (higher score = higher priority for true evaluation):
      - 'cstd': choose largest predictive std (exploration / uncertainty sampling)
      - 'mean': choose smallest predictive mean (pure exploitation)
      - 'cpoi': choose largest Probability of Improvement (uses threshold T)
      - 'cei' : choose largest Expected Improvement (EI)

    Notes:
      - For 'cpoi', threshold is:
            T = f_min - cpoi_eps * (f_max - f_min)
        where f_min/f_max come from the GP training set.
      - This class only does ranking/selection; it does not train the GP.
    """

    def __init__(self, mode: str = "cstd", cpoi_eps: float = 0.05) -> None:
        self.mode = str(mode).lower()
        self.cpoi_eps = float(cpoi_eps)

    def rank(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_train_min: float,
        y_train_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=float).ravel()
        std = np.asarray(std, dtype=float).ravel()
        if mean.shape != std.shape:
            raise ValueError("mean and std must have the same shape")

        crit = self.mode
        if crit == "cstd":
            score = std
        elif crit == "mean":
            score = -mean
        elif crit == "cpoi":
            T = float(y_train_min) - self.cpoi_eps * (float(y_train_max) - float(y_train_min))
            z = (T - mean) / (std + 1e-18)
            score = _normal_cdf(z)
        elif crit == "cei":
            y_best = float(y_train_min)
            z = (y_best - mean) / (std + 1e-18)
            score = (y_best - mean) * _normal_cdf(z) + std * _normal_pdf(z)
        else:
            raise ValueError(f"Unknown DTS surrogate ranking mode: {self.mode}")

        order = np.argsort(-score).astype(int)
        return order, score

    def select(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        y_train_min: float,
        y_train_max: float,
        n_select: int,
    ) -> np.ndarray:
        """Convenience: return top-n indices for true evaluation."""
        order, _ = self.rank(mean, std, y_train_min, y_train_max)
        n = max(1, min(int(n_select), int(order.size)))
        return order[:n]
