import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cma
import numpy as np

from .base import BaseOptimizer


class PyCMAESOptimizer(BaseOptimizer):
    """Wrapper around pycma's CMA-ES with a simple interface."""

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        sigma0: float = 0.5,
        x0: Optional[Sequence[float]] = None,
        popsize: Optional[int] = None,
        popsize_mode: str = "default",  # "default" / "double"
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=dim, seed=seed)
        self._sigma0 = float(sigma0)
        self._x0 = np.asarray(x0 if x0 is not None else np.zeros(dim), dtype=float).reshape(-1)
        if self._x0.size != dim:
            self._x0 = np.resize(self._x0, dim)
        self._popsize_mode = str(popsize_mode).lower()
        opts: Dict[str, Any] = dict(options or {})
        opts.setdefault("seed", seed)
        if "popsize" not in opts:
            if popsize is not None:
                opts["popsize"] = int(popsize)
            else:
                opts["popsize"] = self._resolve_popsize(dim)
        opts.update(kwargs)  # remaining kwargs are treated as CMA options
        # Pop unsupported CMA option if provided
        self._user_callback = opts.pop("callback", None)
        self._options = opts

    def _resolve_popsize(self, dim: int) -> int:
        d = max(2, int(dim))
        if self._popsize_mode == "default":
            return 4 + int(math.floor(3.0 * math.log(d)))
        if self._popsize_mode == "double":
            return 8 + int(math.ceil(6.0 * math.log(d)))
        raise ValueError(f"Unknown popsize_mode: {self._popsize_mode}")

    def optimize(self, problem: Any, max_evals: int) -> Tuple[np.ndarray, float, List[float]]:
        if max_evals <= 0:
            return self._truncate_solution(self._x0), float("inf"), []

        opts = dict(self._options)
        opts["maxfevals"] = min(max_evals, opts.get("maxfevals", max_evals))

        history: List[float] = []

        def record_callback(es: Any) -> None:
            try:
                # Track best f-value per iteration as history.
                history.append(float(es.best.f))
            except Exception:
                pass

        # Combine user callbacks with internal history tracking.
        callbacks: List[Any] = []
        if self._user_callback:
            callbacks.extend(self._user_callback if isinstance(self._user_callback, (list, tuple)) else [self._user_callback])
        callbacks.append(record_callback)

        es = cma.CMAEvolutionStrategy(self._x0, self._sigma0, opts)

        def objective(x: Sequence[float]) -> float:
            return self._evaluate_single(problem, x)

        es.optimize(objective, maxfun=max_evals, callback=callbacks)

        try:
            best_x = np.asarray(es.result.xbest, dtype=float)
            best_f = float(es.result.fbest)
        except Exception:
            best_x = self._x0
            best_f = float("inf")

        if not history:
            history.append(best_f)

        return self._truncate_solution(best_x), float(best_f), history
