import numpy as np
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple


class BaseOptimizer:
    """Shared interface for optimization wrappers."""

    def __init__(self, dim: int, seed: int = 0, **kwargs: Any) -> None:  # noqa: ARG002
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def optimize(self, problem: Any, max_evals: int) -> Tuple[np.ndarray, float, List[float]]:
        """Run the optimizer on ``problem``."""
        raise NotImplementedError

    # Helper utilities
    def _truncate_solution(self, x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            return np.zeros(min(self.dim, 10), dtype=float)
        arr = np.asarray(x, dtype=float).reshape(-1)
        return arr[: min(10, arr.size)]

    def _evaluate_single(self, problem: Any, x: Sequence[float]) -> float:
        """Evaluate one solution using the problem interface."""
        x_arr = np.asarray(x, dtype=float)
        if hasattr(problem, "evaluate") and callable(problem.evaluate):
            val = problem.evaluate(x_arr)
        else:
            val = problem(x_arr)
        arr = np.asarray(val, dtype=float).reshape(-1)
        return float(arr[0] if arr.size else np.nan)

    def _build_batch_evaluator(self, problem: Any) -> Callable[[Iterable[Sequence[float]]], np.ndarray]:
        """Return a function that evaluates a batch of candidates."""
        if hasattr(problem, "evaluate_batch") and callable(problem.evaluate_batch):
            def eval_batch(xs: Iterable[Sequence[float]]) -> np.ndarray:
                vals = problem.evaluate_batch(np.asarray(list(xs), dtype=float))
                return np.asarray(vals, dtype=float).reshape(-1)
            return eval_batch

        def eval_batch(xs: Iterable[Sequence[float]]) -> np.ndarray:
            xs_list = list(xs)
            if not xs_list:
                return np.array([], dtype=float)
            try:
                # Many problems (e.g. Ackley) accept vectorized inputs
                vals = problem.evaluate(np.asarray(xs_list, dtype=float)) if hasattr(problem, "evaluate") else problem(np.asarray(xs_list, dtype=float))
                vals_arr = np.asarray(vals, dtype=float).reshape(-1)
                if vals_arr.size == len(xs_list):
                    return vals_arr
            except Exception:
                pass
            return np.asarray([self._evaluate_single(problem, x) for x in xs_list], dtype=float)

        return eval_batch
