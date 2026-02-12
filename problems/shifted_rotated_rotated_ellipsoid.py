import os
from typing import Optional, Union

import numpy as np


class ShiftedRotatedRotatedEllipsoidProblem:
    """Rotated ellipsoid function with fixed-dimensional shift and rotation."""

    def __init__(
        self,
        dim: int = 10,
        lower_bound: float = -65.536,
        upper_bound: float = 65.536,
        name: str = "shifted_rotated_rotated_ellipsoid",
        optimum: Optional[Union[float, np.ndarray]] = None,
        optimum_path: Optional[str] = None,
        rotation_path: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if dim != 10:
            raise ValueError("ShiftedRotatedRotatedEllipsoidProblem is defined for dim=10.")

        self._dim = dim
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._name = name
        self._rng = rng or np.random.default_rng()

        self._weights = np.arange(dim, 0, -1, dtype=float)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if optimum_path is None:
            optimum_path = os.path.join(base_dir, "rotated_ellipsoid_optimum.txt")
        self._optimum_path = optimum_path

        if rotation_path is None:
            rotation_path = os.path.join(base_dir, "shifted_rotated_rotated_ellipsoid_rotation.txt")
        self._rotation_path = rotation_path

        if optimum is not None:
            opt = self._validate_optimum(optimum)
        else:
            opt = self._load_default_optimum()

        self._optimum = opt
        self._rotation = np.eye(self._dim)

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

    @property
    def rotation(self) -> np.ndarray:
        return self._rotation.copy()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        shifted = x - self._optimum
        return np.sum(self._weights * (shifted**2), axis=1)

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

    def _load_default_optimum(self) -> np.ndarray:
        if os.path.isfile(self._optimum_path):
            try:
                loaded = np.loadtxt(self._optimum_path, ndmin=2)
                if loaded.size >= self._dim:
                    opt = loaded.reshape(-1)[: self._dim]
                    if np.all(opt >= self._lower_bound) and np.all(opt <= self._upper_bound):
                        return opt
            except Exception:
                pass
        opt = self._rng.uniform(self._lower_bound, self._upper_bound, size=self._dim)
        return opt

    def _load_or_create_rotation(self) -> np.ndarray:
        return np.eye(self._dim)
