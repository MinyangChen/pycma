#!/usr/bin/env python
"""Quick check script for HierarchicalSubsystem.

Run from repo root:
    python run_hs5_mod_check.py

It instantiates the problem, evaluates the stored optimum and a random point,
and prints the results (no optimizer run, just a sanity check).
"""

import numpy as np

from problems.hierarchical_subsystem import HierarchicalSubsystem


def main() -> None:
    problem = HierarchicalSubsystem()

    # Evaluate the stored optimum.
    opt_solution = np.concatenate(
        [
            problem._scale_back(problem._griewank_block_first.optimum, problem._griewank_block_first),
            problem._scale_back(problem._griewank_block.optimum, problem._griewank_block),
            problem._scale_back(problem._rosenbrock_block.optimum, problem._rosenbrock_block),
            problem._scale_back(problem._rosenbrock_block.optimum, problem._rosenbrock_block),
            problem._scale_back(problem._ackley_block.optimum, problem._ackley_block),
            problem._scale_back(problem._ackley_block.optimum, problem._ackley_block),
            problem._scale_back(problem._ellipsoid_block.optimum, problem._ellipsoid_block),
            problem._scale_back(problem._ellipsoid_block.optimum, problem._ellipsoid_block),
        ]
    )
    opt_val, opt_detail = problem.evaluate_with_details(opt_solution.reshape(1, -1))

    # Evaluate a random point in the bounds.
    rng = np.random.default_rng(0)
    rand_solution = rng.uniform(problem.lower_bound, problem.upper_bound, size=problem.dim)
    rand_val, _ = problem.evaluate_with_details(rand_solution.reshape(1, -1))

    print("Problem:", problem.name)
    print(f"Dimensions: {problem.dim}, bounds: [{problem.lower_bound}, {problem.upper_bound}]")
    print(f"Optimum fitness (stored optimum): {float(opt_val[0]):.6f}")
    for k, v in opt_detail.items():
        print(f"  {k}: {float(v[0]):.6f}")
    print(f"\nRandom fitness (seed=0): {float(rand_val[0]):.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
