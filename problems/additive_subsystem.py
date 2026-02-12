import numpy as np

from problems.ackley import AckleyProblem
from problems.shifted_rotated_ackley import ShiftedRotatedAckleyProblem
from problems.shifted_rotated_griewank import ShiftedRotatedGriewankProblem
from problems.shifted_rotated_rosenbrock import ShiftedRotatedRosenbrockProblem
from problems.shifted_rotated_schwefel12 import ShiftedRotatedSchwefel12Problem
from problems.shifted_rotated_rotated_ellipsoid import ShiftedRotatedRotatedEllipsoidProblem


class AdditiveSubsystemProblem:
    """
    Composite benchmark formed by ten strongly coupled 10-D subsystems plus a global Ackley interaction.

    Each block is evaluated by its native shifted & rotated function, while a 100-D Ackley term links all variables,
    modelling scenarios where subsystems interact locally but remain weakly dependent across blocks.
    """

    def __init__(self, name: str = "additive_subsystem") -> None:
        self._dim = 100
        self._lower_bound = -32.768
        self._upper_bound = 32.768
        self._name = name

        self._ackley_block = ShiftedRotatedAckleyProblem(dim=10)
        self._griewank_block = ShiftedRotatedGriewankProblem(dim=10)
        self._rosenbrock_block = ShiftedRotatedRosenbrockProblem(dim=10)
        self._schwefel_block = ShiftedRotatedSchwefel12Problem(dim=10)
        self._ellipsoid_block = ShiftedRotatedRotatedEllipsoidProblem(dim=10)
        self._ackley_full = AckleyProblem(dim=100)

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
        x = np.atleast_2d(x)
        if x.shape[1] != self._dim:
            raise ValueError(f"Input dimension must be {self._dim}.")

        blocks = [x[:, i : i + 10] for i in range(0, self._dim, 10)]
        value = np.zeros(x.shape[0], dtype=float)

        value += self._ackley_block(self._scale_block(blocks[0], self._ackley_block)) / 10.0
        value += self._ackley_block(self._scale_block(blocks[1], self._ackley_block)) / 10.0
        value += self._griewank_block(self._scale_block(blocks[2], self._griewank_block)) / 10.0
        value += self._griewank_block(self._scale_block(blocks[3], self._griewank_block)) / 10.0
        value += self._rosenbrock_block(self._scale_block(blocks[4], self._rosenbrock_block)) / 10000.0
        value += self._rosenbrock_block(self._scale_block(blocks[5], self._rosenbrock_block)) / 10000.0
        value += self._schwefel_block(self._scale_block(blocks[6], self._schwefel_block)) / 1000.0
        value += self._schwefel_block(self._scale_block(blocks[7], self._schwefel_block)) / 1000.0
        value += self._ellipsoid_block(self._scale_block(blocks[8], self._ellipsoid_block)) / 1000.0
        value += self._ellipsoid_block(self._scale_block(blocks[9], self._ellipsoid_block)) / 1000.0

        # Global Ackley term across all 100 variables
        scaled_full = self._rescale(
            x,
            self._lower_bound,
            self._upper_bound,
            self._ackley_full.lower_bound,
            self._ackley_full.upper_bound,
        )
        value += self._ackley_full(scaled_full) / 20.0

        return value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def evaluate_with_details(self, x: np.ndarray):
        x = np.atleast_2d(x)
        if x.shape[1] != self._dim:
            raise ValueError(f"Input dimension must be {self._dim}.")

        blocks = [x[:, i : i + 10] for i in range(0, self._dim, 10)]

        subvalues = []
        subvalues.append(self._ackley_block(self._scale_block(blocks[0], self._ackley_block)) / 10.0)
        subvalues.append(self._ackley_block(self._scale_block(blocks[1], self._ackley_block)) / 10.0)
        subvalues.append(self._griewank_block(self._scale_block(blocks[2], self._griewank_block)) / 10.0)
        subvalues.append(self._griewank_block(self._scale_block(blocks[3], self._griewank_block)) / 10.0)
        subvalues.append(self._rosenbrock_block(self._scale_block(blocks[4], self._rosenbrock_block)) / 10000.0)
        subvalues.append(self._rosenbrock_block(self._scale_block(blocks[5], self._rosenbrock_block)) / 10000.0)
        subvalues.append(self._schwefel_block(self._scale_block(blocks[6], self._schwefel_block)) / 1000.0)
        subvalues.append(self._schwefel_block(self._scale_block(blocks[7], self._schwefel_block)) / 1000.0)
        subvalues.append(self._ellipsoid_block(self._scale_block(blocks[8], self._ellipsoid_block)) / 1000.0)
        subvalues.append(self._ellipsoid_block(self._scale_block(blocks[9], self._ellipsoid_block)) / 1000.0)

        scaled_full = self._rescale(
            x,
            self._lower_bound,
            self._upper_bound,
            self._ackley_full.lower_bound,
            self._ackley_full.upper_bound,
        )
        global_ackley = self._ackley_full(scaled_full) / 20.0

        total = np.zeros(x.shape[0], dtype=float)
        for val in subvalues:
            total += val
        total += global_ackley

        return total, {
            "block_1": subvalues[0],
            "block_2": subvalues[1],
            "block_3": subvalues[2],
            "block_4": subvalues[3],
            "block_5": subvalues[4],
            "block_6": subvalues[5],
            "block_7": subvalues[6],
            "block_8": subvalues[7],
            "block_9": subvalues[8],
            "block_10": subvalues[9],
            "global_ackley": global_ackley,
        }

        # return total, [
        #     subvalues[0],
        #     subvalues[1],
        #     subvalues[2],
        #     subvalues[3],
        #     subvalues[4],
        #     subvalues[5],
        #     subvalues[6],
        #     subvalues[7],
        #     subvalues[8],
        #     subvalues[9],
        #     global_ackley,
        # ]

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


if __name__ == "__main__":
    problem = AdditiveSubsystemProblem()
    design = np.array(
        [
            -12.104,
            26.139,
            -10.032,
            10.87,
            20.134,
            -29.14,
            -8.128,
            -29.66,
            2.054,
            -5.694,
            3.003,
            -9.854,
            -20.233,
            26.803,
            18.846,
            3.004,
            9.07,
            -22.598,
            9.138,
            -6.547,
            7.858,
            6.649,
            11.344,
            8.848,
            -19.239,
            6.211,
            -13.528,
            5.246,
            -30.547,
            22.847,
            11.042,
            10.755,
            8.875,
            8.936,
            -21.556,
            6.727,
            -10.901,
            8.263,
            -21.617,
            25.739,
            30.822,
            27.828,
            -25.408,
            10.237,
            -2.997,
            -0.969,
            -21.548,
            -7.563,
            -14.165,
            -9.304,
            23.355,
            15.013,
            -11.907,
            10.142,
            -6.372,
            -17.873,
            -24.124,
            1.596,
            -11.093,
            4.044,
            10.907,
            -27.941,
            26.247,
            7.42,
            -6.62,
            4.015,
            -12.038,
            -2.77,
            8.661,
            21.454,
            -1.79,
            -16.95,
            17.374,
            10.028,
            -5.607,
            -0.556,
            5.72,
            -8.428,
            -11.219,
            19.15,
            5.837,
            -23.385,
            29.483,
            -9.426,
            -26.084,
            4.726,
            -15.335,
            28.788,
            -25.514,
            24.886,
            3.691,
            -20.452,
            31.869,
            -9.936,
            -26.831,
            -1.009,
            -21.372,
            29.365,
            -27.561,
            25.087,
        ]
    )
    total, details = problem.evaluate_with_details(design.reshape(1, -1))
    print(f"Total fitness: {float(total[0]):.6f}")
    for key, value in details.items():
        print(f"{key}: {float(value[0]):.6f}")
