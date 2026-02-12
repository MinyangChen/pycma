import os
from typing import Optional, Union

import numpy as np


class SlightlyRotatedEllipsoidProblem:
    """Ellipsoid with moderate conditioning and only a mild random rotation."""

    def __init__(
        self,
        dim: int = 10,
        lower_bound: float = -100.0,
        upper_bound: float = 100.0,
        name: str = "slightly_rotated_ellipsoid",
        optimum: Optional[Union[float, np.ndarray]] = None,
        optimum_path: Optional[str] = None,
        rotation_path: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._dim = dim
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._name = name
        self._rng = rng or np.random.default_rng()
        self._condition = 30.0
        self._weights = np.geomspace(1.0, self._condition, num=self._dim)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if optimum_path is None:
            optimum_path = os.path.join(base_dir, "slightly_rotated_ellipsoid_optimum.txt")
        self._optimum_path = optimum_path

        if rotation_path is None:
            rotation_path = os.path.join(base_dir, "slightly_rotated_ellipsoid_rotation.txt")
        self._rotation_path = rotation_path

        if optimum is not None:
            opt = self._validate_optimum(optimum)
            self._optimum = opt
            self._save_optimum(opt)
        else:
            self._optimum = self._load_or_create_optimum()

        self._rotation = self._load_or_create_rotation()

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
        rotated = shifted @ self._rotation
        return np.sum(self._weights * rotated**2, axis=1)

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
                if loaded.shape[1] >= self._dim:
                    candidate = loaded.reshape(-1)[: self._dim]
                    if np.all(candidate >= self._lower_bound) and np.all(candidate <= self._upper_bound):
                        return candidate
            except Exception:
                pass
        opt = self._rng.uniform(-5.0, 5.0, size=self._dim)
        self._save_optimum(opt)
        return opt

    def _save_optimum(self, optimum: np.ndarray) -> None:
        os.makedirs(os.path.dirname(self._optimum_path), exist_ok=True)
        np.savetxt(self._optimum_path, optimum[None, :], fmt="%.16f")

    def _load_or_create_rotation(self) -> np.ndarray:
        if os.path.isfile(self._rotation_path):
            try:
                loaded = np.loadtxt(self._rotation_path)
                if loaded.size == self._dim * self._dim:
                    matrix = loaded.reshape(self._dim, self._dim)
                    return matrix
            except Exception:
                pass

        rotation = self._generate_slight_rotation()
        self._save_rotation(rotation)
        return rotation

    def _generate_slight_rotation(self) -> np.ndarray:
        epsilon = 0.05
        perturb = np.eye(self._dim) + epsilon * self._rng.standard_normal((self._dim, self._dim))
        q, _ = np.linalg.qr(perturb)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1.0
        return q

    def _save_rotation(self, rotation: np.ndarray) -> None:
        os.makedirs(os.path.dirname(self._rotation_path), exist_ok=True)
        np.savetxt(self._rotation_path, rotation, fmt="%.16f")
