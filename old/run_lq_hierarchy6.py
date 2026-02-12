#!/usr/bin/env python
"""Run lq-CMA-ES on the HierarchySubsystemProblem6 benchmark.

Run from repo root (uses local ``cma`` and ``problems``):
    python run_lq_hierarchy6.py
Controls (simple):
- set ``run_times`` below (seeds = 0..run_times-1, or override via CMA_SEEDS env to a comma-list)
- tweak ``maxfevals`` for budget per run
"""

import os
import sys
import time
from typing import Sequence

try:
    import numpy as np  # noqa: F401
except ImportError as exc:
    sys.exit("numpy is required. Install with `python -m pip install numpy`: %s" % exc)

import cma
from problems.hierarchy_subsystem_6 import HierarchySubsystemProblem6


def main() -> None:
    if not hasattr(cma, "fmin_lq_surr2"):
        sys.exit("cma.fmin_lq_surr2 not found; ensure you're using this repo's `cma` package.")

    problem = HierarchySubsystemProblem6()
    dim = problem.dim

    # Start near zero inside the box bounds.
    x0: Sequence[float] = [0.0] * dim
    sigma0 = (problem.upper_bound - problem.lower_bound) / 10.0  # moderate initial step size

    run_times = int(os.environ.get("RUN_TIMES", 1))
    maxfevals = int(os.environ.get("MAXFEVALS", 10000))

    seeds_env = os.environ.get("CMA_SEEDS")
    seeds = [int(s) for s in seeds_env.split(",")] if seeds_env else list(range(run_times))

    def objective(x) -> float:
        """Wrap the problem to return a scalar float."""
        return float(problem(np.asarray(x)))

    def run_once(seed: int):
        opts = {
            "seed": seed,
            "bounds": [problem.lower_bound, problem.upper_bound],
            "maxfevals": maxfevals,
            "verb_disp": 200,
        }
        t0 = time.time()
        _, es = cma.fmin_lq_surr2(objective, x0, sigma0, opts, incpopsize=2)
        elapsed = time.time() - t0
        return {
            "seed": seed,
            "fbest": es.result.fbest,
            "evals": es.result.evaluations,
            "iters": es.countiter,
            "stop": es.stop(),
            "elapsed": elapsed,
        }

    results = [run_once(seed) for seed in seeds]

    print("\n=== lq-CMA-ES on HierarchySubsystemProblem6 (multiple runs) ===")
    for r in results:
        print(
            f"seed={r['seed']:>3} | fbest={r['fbest']:.4e} | "
            f"evals={r['evals']} | iters={r['iters']} | stop={list(r['stop'].keys())} | "
            f"time={r['elapsed']:.2f}s"
        )

    best = min(results, key=lambda r: r["fbest"])
    print(
        f"\nBest run: seed={best['seed']} with fbest={best['fbest']:.4e} "
        f"after {best['evals']} evals (stop={list(best['stop'].keys())}, time={best['elapsed']:.2f}s)"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
