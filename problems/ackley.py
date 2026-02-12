import os
from typing import Optional, Union

import numpy as np


class AckleyProblem:
    """Ackley benchmark optimisation problem with a stored random optimum."""

    def __init__(
        self,
        dim: int = 100,
        lower_bound: Union[float, np.ndarray] = -32.768,
        upper_bound: Union[float, np.ndarray] = 32.768,
        name: str = "ackley",
        optimum: Optional[Union[float, np.ndarray]] = None,
        optimum_path: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._dim = dim
        lb = np.asarray(lower_bound, dtype=float)
        ub = np.asarray(upper_bound, dtype=float)
        if lb.size == 1:
            # Expand scalar bounds to per-dimension vectors.
            lb = np.full(dim, float(lb))
        if ub.size == 1:
            ub = np.full(dim, float(ub))
        self._lower_bound = lb.reshape(dim)
        self._upper_bound = ub.reshape(dim)
        self._name = name
        self._rng = rng or np.random.default_rng()

        if optimum_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            optimum_path = os.path.join(base_dir, "ackley_optimum.txt")
        self._optimum_path = optimum_path

        if optimum is not None:
            opt = self._validate_optimum(optimum)
            self._optimum = opt
            self._save_optimum(opt)
        else:
            self._optimum = self._load_or_create_optimum()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def lower_bound(self) -> np.ndarray:
        return self._lower_bound.copy()

    @property
    def upper_bound(self) -> np.ndarray:
        return self._upper_bound.copy()

    @property
    def name(self) -> str:
        return self._name

    @property
    def optimum(self) -> np.ndarray:
        return self._optimum.copy()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Vectorized Ackley function shifted to a stored optimum."""
        a = 20.0
        b = 0.2
        c = 2.0 * np.pi
        x = np.atleast_2d(x)
        shifted = x - self._optimum
        # shifted = x
        term1 = -a * np.exp(-b * np.sqrt(np.mean(shifted**2, axis=1)))
        term2 = -np.exp(np.mean(np.cos(c * shifted), axis=1))
        return term1 + term2 + a + np.e

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def _validate_optimum(self, optimum: Union[float, np.ndarray]) -> np.ndarray:
        opt = np.asarray(optimum, dtype=float)
        if opt.size == 1:
            opt = np.full(self._dim, float(opt))
        opt = opt.reshape(self._dim)
        if np.any(opt < self._lower_bound) or np.any(opt > self._upper_bound):
            raise ValueError("Optimum must lie within the problem bounds.")
        return opt

    def _load_or_create_optimum(self) -> np.ndarray:
        if os.path.isfile(self._optimum_path):
            try:
                loaded = np.loadtxt(self._optimum_path, ndmin=2)
                if loaded.shape[1] == self._dim:
                    opt = loaded.reshape(self._dim)
                    if np.all(opt >= self._lower_bound) and np.all(opt <= self._upper_bound):
                        return opt
            except Exception:
                pass

        opt = self._rng.uniform(self._lower_bound, self._upper_bound, size=self._dim)
        opt = np.clip(opt, self._lower_bound, self._upper_bound)
        self._save_optimum(opt)
        return opt

    def _save_optimum(self, optimum: np.ndarray) -> None:
        os.makedirs(os.path.dirname(self._optimum_path), exist_ok=True)
        np.savetxt(self._optimum_path, optimum[None, :], fmt="%.16f")
