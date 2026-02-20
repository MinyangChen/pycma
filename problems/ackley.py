import os
from typing import Optional, Union

import numpy as np


class AckleyProblem:
    """Ackley benchmark optimisation problem with a fixed stored optimum."""

    _OPTIMUM_LENGTH = 100

    def __init__(
        self,
        dim: int = 100,
        lower_bound: Union[float, np.ndarray] = -32.768,
        upper_bound: Union[float, np.ndarray] = 32.768,
        name: str = "ackley",
        optimum_path: Optional[str] = None,
    ) -> None:
        if dim > self._OPTIMUM_LENGTH:
            raise ValueError(f"dim={dim} exceeds optimum length ({self._OPTIMUM_LENGTH}).")

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

        if optimum_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            opt_dir = os.path.join(base_dir, "optimum")
            optimum_path = os.path.join(opt_dir, "ackley_optimum_100d.txt")
        self._optimum_path = optimum_path

        full_optimum = self._load_fixed_optimum()
        if dim < self._OPTIMUM_LENGTH:
            opt = full_optimum[:dim]
        else:
            opt = full_optimum
        self._optimum = self._validate_optimum(opt)

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

    def _load_fixed_optimum(self) -> np.ndarray:
        if not os.path.isfile(self._optimum_path):
            raise FileNotFoundError(
                f"Ackley optimum file not found at '{self._optimum_path}'."
            )
        try:
            loaded = np.loadtxt(self._optimum_path, ndmin=2, dtype=float)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse Ackley optimum file at '{self._optimum_path}': {exc}"
            ) from exc

        if loaded.shape != (1, self._OPTIMUM_LENGTH):
            raise ValueError(
                f"Ackley optimum file at '{self._optimum_path}' must contain exactly "
                f"1 row and {self._OPTIMUM_LENGTH} columns, got shape {loaded.shape}."
            )
        if not np.all(np.isfinite(loaded)):
            raise ValueError(
                f"Ackley optimum file at '{self._optimum_path}' contains non-finite values."
            )
        return loaded.reshape(self._OPTIMUM_LENGTH)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    opt_dir = os.path.join(base_dir, "optimum")
    opt_path = os.path.join(opt_dir, "ackley_optimum_100d.txt")

    if os.path.exists(opt_path):
        print(f"Ackley optimum file already exists at '{opt_path}'. Not overwriting.")
    else:
        os.makedirs(opt_dir, exist_ok=True)
        rng = np.random.default_rng(42)
        lower_bound = -32.768
        upper_bound = 32.768
        optimum = rng.uniform(lower_bound, upper_bound, size=100)
        np.savetxt(opt_path, optimum[None, :], fmt="%.16f")
        print(f"Created Ackley optimum file at '{opt_path}'.")
