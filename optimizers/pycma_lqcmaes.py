from typing import Any, Dict, List, Optional, Sequence, Tuple

import cma
import numpy as np

from .base import BaseOptimizer


class PyLQCMAESOptimizer(BaseOptimizer):
    """Wrapper around pycma's lq-CMA-ES (surrogate-assisted)."""

    def __init__(
        self,
        dim: int,
        seed: int = 0,
        sigma0: float = 0.5,
        x0: Optional[Sequence[float]] = None,
        options: Optional[Dict[str, Any]] = None,
        inject: bool = True,
        restarts: int = 0,
        incpopsize: int = 2,
        keep_model: bool = False,
        callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(dim=dim, seed=seed)
        self._sigma0 = float(sigma0)
        self._x0 = np.asarray(x0 if x0 is not None else np.zeros(dim), dtype=float).reshape(-1)
        if self._x0.size != dim:
            self._x0 = np.resize(self._x0, dim)

        opts: Dict[str, Any] = dict(options or {})
        opts.setdefault("seed", seed)
        opts.update(kwargs)  # remaining kwargs are CMA options
        self._options = opts

        self._inject = inject
        self._restarts = restarts
        self._incpopsize = incpopsize
        self._keep_model = keep_model
        self._user_callback = callback

    def optimize(self, problem: Any, max_evals: int) -> Tuple[np.ndarray, float, List[float]]:
        if not hasattr(cma, "fmin_lq_surr2"):
            raise RuntimeError("pycma with lq-CMA-ES support is required (fmin_lq_surr2 missing).")

        history: List[float] = []

        def objective(x: Sequence[float]) -> float:
            return self._evaluate_single(problem, x)

        def combined_callback(es: Any) -> None:
            if self._user_callback:
                self._user_callback(es)
            try:
                # Use true-evaluation best (xfavorite) for a stable history.
                if hasattr(es, "best") and es.best.f is not None:
                    history.append(float(es.best.f))
            except Exception:
                pass

        opts = dict(self._options)
        opts["maxfevals"] = min(max_evals, opts.get("maxfevals", max_evals))

        xbest, es = cma.fmin_lq_surr2(
            objective,
            self._x0,
            self._sigma0,
            opts,
            inject=self._inject,
            restarts=self._restarts,
            incpopsize=self._incpopsize,
            keep_model=self._keep_model,
            callback=combined_callback,
        )

        best_x: Optional[np.ndarray] = None
        best_f: float = float("inf")

        if hasattr(es, "best") and getattr(es.best, "x", None) is not None:
            best_x = np.asarray(es.best.x, dtype=float)
            try:
                best_f = float(es.best.f)
            except Exception:
                best_f = float("inf")

        if best_x is None and hasattr(es, "result"):
            try:
                best_x = np.asarray(es.result.xbest, dtype=float)
                best_f = float(es.result.fbest)
            except Exception:
                pass

        if best_x is None:
            best_x = np.asarray(xbest, dtype=float)
            best_f = float(self._evaluate_single(problem, best_x))

        if not history and best_f != float("inf"):
            history.append(best_f)

        return self._truncate_solution(best_x), float(best_f), history
