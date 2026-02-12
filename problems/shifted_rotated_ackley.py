import os
from typing import Optional, Union

import numpy as np


class ShiftedRotatedAckleyProblem:
    """Ackley function with fixed-dimensional shift and rotation."""

    def __init__(
        self,
        dim: int = 10,
        lower_bound: float = -32.768,
        upper_bound: float = 32.768,
        name: str = "shifted_rotated_ackley",
        optimum: Optional[Union[float, np.ndarray]] = None,
        optimum_path: Optional[str] = None,
        rotation_path: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if dim != 10:
            raise ValueError("ShiftedRotatedAckleyProblem is defined for dim=10.")

        self._dim = dim
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._name = name
        self._rng = rng or np.random.default_rng()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        if optimum_path is None:
            optimum_path = os.path.join(base_dir, "ackley_optimum.txt")
        self._optimum_path = optimum_path

        if rotation_path is None:
            rotation_path = os.path.join(base_dir, "shifted_rotated_ackley_rotation.txt")
        self._rotation_path = rotation_path

        if optimum is not None:
            opt = self._validate_optimum(optimum)
        else:
            opt = self._load_default_optimum()

        self._optimum = opt
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
        """Ackley evaluation after shifting and rotating inputs."""
        a = 20.0
        b = 0.2
        c = 2.0 * np.pi
        x = np.atleast_2d(x)
        shifted = x - self._optimum
        rotated = shifted @ self._rotation
        term1 = -a * np.exp(-b * np.sqrt(np.mean(rotated**2, axis=1)))
        term2 = -np.exp(np.mean(np.cos(c * rotated), axis=1))
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
        if os.path.isfile(self._rotation_path):
            try:
                loaded = np.loadtxt(self._rotation_path)
                if loaded.shape == (self._dim, self._dim):
                    return loaded
            except Exception:
                pass

        random_matrix = self._rng.standard_normal((self._dim, self._dim))
        q, r = np.linalg.qr(random_matrix)
        diag = np.sign(np.diag(r))
        diag[diag == 0] = 1.0
        rotation = q * diag

        os.makedirs(os.path.dirname(self._rotation_path), exist_ok=True)
        np.savetxt(self._rotation_path, rotation, fmt="%.16f")
        return rotation
