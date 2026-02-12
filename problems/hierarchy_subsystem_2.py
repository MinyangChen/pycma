import numpy as np

from problems.shifted_rotated_ackley import ShiftedRotatedAckleyProblem
from problems.shifted_rotated_griewank import ShiftedRotatedGriewankProblem
from problems.shifted_rotated_rosenbrock import ShiftedRotatedRosenbrockProblem
from problems.shifted_rotated_schwefel12 import ShiftedRotatedSchwefel12Problem
from problems.shifted_rotated_rotated_ellipsoid import ShiftedRotatedRotatedEllipsoidProblem


class HierarchySubsystemProblem2:
    """Hierarchy benchmark linking ten shifted/rotated subsystems via a top-level Griewank interaction."""

    def __init__(self, name: str = "hierarchy_subsystem_2") -> None:
        self._dim = 100
        self._lower_bound = -32.768
        self._upper_bound = 32.768
        self._name = name

        self._ackley_block = ShiftedRotatedAckleyProblem(dim=10)
        self._griewank_block = ShiftedRotatedGriewankProblem(dim=10)
        self._rosenbrock_block = ShiftedRotatedRosenbrockProblem(dim=10)
        self._schwefel_block = ShiftedRotatedSchwefel12Problem(dim=10)
        self._ellipsoid_block = ShiftedRotatedRotatedEllipsoidProblem(dim=10)
        self._top_griewank = ShiftedRotatedGriewankProblem(dim=10)

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

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        total, _ = self.evaluate_with_details(x)
        return total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def evaluate_with_details(self, x: np.ndarray):
        x = np.atleast_2d(x)
        if x.shape[1] != self._dim:
            raise ValueError(f"Input dimension must be {self._dim}.")

        chunks = [x[:, i : i + 10] for i in range(0, self._dim, 10)]

        z_vals = []
        detail = {}
        top_opt = self._top_griewank.optimum

        def block_value(block, problem, label, idx):
            vals = problem(block)
            detail[label] = vals
            return np.log(vals + 1.0) + top_opt[idx]

        z_vals.append(block_value(self._scale_block(chunks[0], self._ackley_block), self._ackley_block, "ShiftedRotatedAckley_1", 0))
        z_vals.append(block_value(self._scale_block(chunks[1], self._ackley_block), self._ackley_block, "ShiftedRotatedAckley_2", 1))
        z_vals.append(block_value(self._scale_block(chunks[2], self._griewank_block), self._griewank_block, "ShiftedRotatedGriewank_1_Block", 2))
        z_vals.append(block_value(self._scale_block(chunks[3], self._griewank_block), self._griewank_block, "ShiftedRotatedGriewank_2_Block", 3))
        z_vals.append(block_value(self._scale_block(chunks[4], self._rosenbrock_block), self._rosenbrock_block, "ShiftedRotatedRosenbrock_1", 4))
        z_vals.append(block_value(self._scale_block(chunks[5], self._rosenbrock_block), self._rosenbrock_block, "ShiftedRotatedRosenbrock_2", 5))
        z_vals.append(block_value(self._scale_block(chunks[6], self._schwefel_block), self._schwefel_block, "ShiftedRotatedSchwefel12_1", 6))
        z_vals.append(block_value(self._scale_block(chunks[7], self._schwefel_block), self._schwefel_block, "ShiftedRotatedSchwefel12_2", 7))
        z_vals.append(block_value(self._scale_block(chunks[8], self._ellipsoid_block), self._ellipsoid_block, "ShiftedRotatedEllipsoid_1", 8))
        z_vals.append(block_value(self._scale_block(chunks[9], self._ellipsoid_block), self._ellipsoid_block, "ShiftedRotatedEllipsoid_2", 9))

        z = np.stack(z_vals, axis=1)
        top_value = self._top_griewank(z)
        detail["Top_ShiftedRotatedGriewank"] = top_value

        return top_value, detail

    def _scale_block(self, block: np.ndarray, subproblem) -> np.ndarray:
        return self._rescale(
            block,
            self._lower_bound,
            self._upper_bound,
            subproblem.lower_bound,
            subproblem.upper_bound,
        )

    @staticmethod
    def _rescale(values: np.ndarray, src_low: float, src_high: float, tgt_low: float, tgt_high: float) -> np.ndarray:
        if src_high == src_low:
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
    problem = HierarchySubsystemProblem2()

    opt_solution = np.concatenate([
        problem._scale_back(problem._ackley_block.optimum, problem._ackley_block),
        problem._scale_back(problem._ackley_block.optimum, problem._ackley_block),
        problem._scale_back(problem._griewank_block.optimum, problem._griewank_block),
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
