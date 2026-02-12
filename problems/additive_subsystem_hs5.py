import os
import numpy as np

from problems.shifted_rotated_griewank_flexible import ShiftedRotatedGriewankFlexible
from problems.shifted_rotated_rosenbrock import ShiftedRotatedRosenbrockProblem
from problems.shifted_rotated_schwefel12 import ShiftedRotatedSchwefel12Problem
from problems.shifted_rotated_rotated_ellipsoid import ShiftedRotatedRotatedEllipsoidProblem


class AdditiveSubsystem:
    """
    Additive variant of HierarchicalSubsystem: sum sub-block objectives (no top function).
    Uses the same subproblems and scaling as HierarchicalSubsystem.
    """

    def __init__(self, name: str = "additive_subsystem") -> None:
        self._dim = 100
        # Keep bounds per-dimension for consistent scaling.
        lb = np.full(self._dim, -32.768, dtype=float)
        ub = np.full(self._dim, 32.768, dtype=float)
        self._lower_bound = lb
        self._upper_bound = ub
        self._name = name

        # Same subproblems as the base mod version.
        self._griewank_block_first = ShiftedRotatedGriewankFlexible(dim=20)
        self._griewank_block = ShiftedRotatedGriewankFlexible(dim=20)
        self._rosenbrock_block = ShiftedRotatedRosenbrockProblem(dim=10)
        self._schwefel_block = ShiftedRotatedSchwefel12Problem(dim=10)
        self._ellipsoid_block = ShiftedRotatedRotatedEllipsoidProblem(dim=10)

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

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        total, _ = self.evaluate_with_details(x)
        return total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def evaluate_with_details(self, x: np.ndarray):
        x = np.atleast_2d(x)
        if x.shape[1] != self._dim:
            raise ValueError(f"Input dimension must be {self._dim}.")

        chunk_sizes = [20, 20, 10, 10, 10, 10, 10, 10]
        chunks = []
        start = 0
        for size in chunk_sizes:
            chunks.append(x[:, start : start + size])
            start += size

        detail = {}
        vals = []

        def add_block(block, problem, label):
            v = problem(block)
            detail[label] = v
            vals.append(v)

        # add_block(self._scale_block(chunks[0], self._griewank_block_first), self._griewank_block_first, "ShiftedRotatedGriewank_1_20d")
        # add_block(self._scale_block(chunks[1], self._griewank_block), self._griewank_block, "ShiftedRotatedGriewank_2_20d")
        # add_block(self._scale_block(chunks[2], self._rosenbrock_block), self._rosenbrock_block, "ShiftedRotatedRosenbrock_1")
        # add_block(self._scale_block(chunks[3], self._rosenbrock_block), self._rosenbrock_block, "ShiftedRotatedRosenbrock_2")
        # add_block(self._scale_block(chunks[4], self._schwefel_block), self._schwefel_block, "ShiftedRotatedSchwefel12_1")
        # add_block(self._scale_block(chunks[5], self._schwefel_block), self._schwefel_block, "ShiftedRotatedSchwefel12_2")
        # add_block(self._scale_block(chunks[6], self._ellipsoid_block), self._ellipsoid_block, "ShiftedRotatedEllipsoid_1")
        # add_block(self._scale_block(chunks[7], self._ellipsoid_block), self._ellipsoid_block, "ShiftedRotatedEllipsoid_2")

        add_block(self._scale_block(chunks[0], self._griewank_block_first), self._griewank_block_first, "ShiftedRotatedGriewank_1_20d")
        add_block(self._scale_block(chunks[1], self._griewank_block), self._griewank_block, "ShiftedRotatedGriewank_2_20d")
        add_block(self._scale_block(chunks[2], self._schwefel_block), self._schwefel_block, "ShiftedRotatedSchwefel12_1")
        add_block(self._scale_block(chunks[3], self._schwefel_block), self._schwefel_block, "ShiftedRotatedSchwefel12_2")
        add_block(self._scale_block(chunks[4], self._schwefel_block), self._schwefel_block, "ShiftedRotatedSchwefel12_3")
        add_block(self._scale_block(chunks[5], self._ellipsoid_block), self._ellipsoid_block, "ShiftedRotatedEllipsoid_1")
        add_block(self._scale_block(chunks[6], self._ellipsoid_block), self._ellipsoid_block, "ShiftedRotatedEllipsoid_2")
        add_block(self._scale_block(chunks[7], self._ellipsoid_block), self._ellipsoid_block, "ShiftedRotatedEllipsoid_3")

        total = sum(vals)
        return total, detail

    def _scale_block(self, block: np.ndarray, subproblem) -> np.ndarray:
        # use matching-length bounds for broadcasting with the block
        src_low = self._lower_bound[: block.shape[1]] if self._lower_bound.size > block.shape[1] else self._lower_bound
        src_high = self._upper_bound[: block.shape[1]] if self._upper_bound.size > block.shape[1] else self._upper_bound
        return self._rescale(
            block,
            src_low,
            src_high,
            subproblem.lower_bound,
            subproblem.upper_bound,
        )

    @staticmethod
    def _rescale(values: np.ndarray, src_low, src_high, tgt_low, tgt_high) -> np.ndarray:
        src_low = np.asarray(src_low, dtype=float)
        src_high = np.asarray(src_high, dtype=float)
        tgt_low = np.asarray(tgt_low, dtype=float)
        tgt_high = np.asarray(tgt_high, dtype=float)
        if np.all(src_high == src_low):
            raise ValueError("Source range has zero width; cannot rescale.")
        scale = (tgt_high - tgt_low) / (src_high - src_low)
        return tgt_low + (values - src_low) * scale

    def _scale_back(self, values: np.ndarray, subproblem) -> np.ndarray:
        return self._rescale(
            values,
            subproblem.lower_bound,
            subproblem.upper_bound,
            self._lower_bound,
            self._upper_bound,
        )


if __name__ == "__main__":
    problem = AdditiveSubsystem()

    opt_solution = np.concatenate([
        problem._scale_back(problem._griewank_block_first.optimum, problem._griewank_block_first),
        problem._scale_back(problem._griewank_block.optimum, problem._griewank_block),
        problem._scale_back(problem._rosenbrock_block.optimum, problem._rosenbrock_block),
        problem._scale_back(problem._rosenbrock_block.optimum, problem._rosenbrock_block),
        problem._scale_back(problem._schwefel_block.optimum, problem._schwefel_block),
        problem._scale_back(problem._schwefel_block.optimum, problem._schwefel_block),
        problem._scale_back(problem._ellipsoid_block.optimum, problem._ellipsoid_block),
        problem._scale_back(problem._ellipsoid_block.optimum, problem._ellipsoid_block),
    ])

    total_opt, details_opt = problem.evaluate_with_details(opt_solution.reshape(1, -1))
    print("Optimal solution fitness:", float(total_opt[0]))
    for key, value in details_opt.items():
        print(f"{key}: {float(value[0]):.6f}")

    rng = np.random.default_rng(42)
    rand_solution = rng.uniform(problem.lower_bound, problem.upper_bound, size=problem.dim)
    total_rand, details_rand = problem.evaluate_with_details(rand_solution.reshape(1, -1))
    print("\nRandom solution fitness:", float(total_rand[0]))
    for key, value in details_rand.items():
        print(f"{key}: {float(value[0]):.6f}")
