import os
from typing import Optional, Union

import numpy as np


class Schwefel12Problem:
    """Schwefel 1.2 benchmark optimisation problem with a stored random optimum."""

    def __init__(
        self,
        dim: int = 100,
        lower_bound: float = -100.0,
        upper_bound: float = 100.0,
        name: str = "schwefel_1_2",
        optimum: Optional[Union[float, np.ndarray]] = None,
        optimum_path: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._dim = dim
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._name = name
        self._rng = rng or np.random.default_rng()

        if optimum_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            optimum_path = os.path.join(base_dir, "schwefel12_optimum.txt")
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
    def lower_bound(self) -> float:
        return self._lower_bound

    @property
    def upper_bound(self) -> float:
        return self._upper_bound

    @property
    def name(self) -> str:
        return self._name

    @property
    def optimum(self) -> np.ndarray:
        return self._optimum.copy()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized Schwefel 1.2 function shifted to a stored optimum.
        """
        x = np.atleast_2d(x)
        shifted = x - self._optimum
        cumulative = np.cumsum(shifted, axis=1)
        return np.sum(cumulative**2, axis=1)

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
        self._save_optimum(opt)
        return opt

    def _save_optimum(self, optimum: np.ndarray) -> None:
        os.makedirs(os.path.dirname(self._optimum_path), exist_ok=True)
        np.savetxt(self._optimum_path, optimum[None, :], fmt="%.16f")
