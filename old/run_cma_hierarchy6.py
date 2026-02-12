#!/usr/bin/env python
"""Run classic CMA-ES on the HierarchySubsystemProblem6 benchmark.

Usage (from repo root, uses local ``cma`` and ``problems``):
    python run_cma_hierarchy6.py
    python run_cma_hierarchy6.py optimize   # uses the OOOptimizer.optimize helper

Controls (simple):
- set ``run_times`` below (seeds = 0..run_times-1, or override with CMA_SEEDS env)
- tweak ``maxfevals`` for budget per run (env MAXFEVALS overrides)
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
# from problems.hierarchy_subsystem_6 import HierarchySubsystemProblem6
from problems.hierarchical_subsystem import HierarchicalSubsystem
from problems.additive_subsystem_hs5 import AdditiveSubsystem

def main() -> None:
    if not hasattr(cma, "CMAEvolutionStrategy"):
        sys.exit("CMAEvolutionStrategy not found; ensure you're using this repo's `cma` package.")

    problem = HierarchicalSubsystem()
    dim = problem.dim

    x0: Sequence[float] = [0.0] * dim  # start inside the bounds
    sigma0 = (problem.upper_bound - problem.lower_bound) / 10.0  # moderate initial step size

    run_times = int(os.environ.get("RUN_TIMES", 31))
    maxfevals = int(os.environ.get("MAXFEVALS", 100000))

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
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        t0 = time.time()
        while not es.stop():
            candidates = es.ask()
            es.tell(candidates, [objective(x) for x in candidates])
            es.logger.add()
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

    print("\n=== CMA-ES on HierarchySubsystemProblem6 (multiple runs) ===")
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


def main_optimize() -> None:
    """Same experiment but via the OOOptimizer.optimize helper (no manual ask/tell loop)."""
    if not hasattr(cma, "CMAEvolutionStrategy"):
        sys.exit("CMAEvolutionStrategy not found; ensure you're using this repo's `cma` package.")

    # problem = HierarchicalSubsystem()
    problem = AdditiveSubsystem()
    dim = problem.dim

    x0: Sequence[float] = [0.0] * dim
    sigma0 = (problem.upper_bound - problem.lower_bound) / 10.0

    run_times = int(os.environ.get("RUN_TIMES", 31))
    maxfevals = int(os.environ.get("MAXFEVALS", 70000))

    seeds_env = os.environ.get("CMA_SEEDS")
    seeds = [int(s) for s in seeds_env.split(",")] if seeds_env else list(range(run_times))

    def objective(x) -> float:
        return float(problem(np.asarray(x)))

    def run_once(seed: int):
        opts = {
            "seed": seed,
            "bounds": [problem.lower_bound, problem.upper_bound],
            "maxfevals": maxfevals,
            "verb_disp": 200,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        t0 = time.time()
        # `optimize` runs the ask/tell loop internally; logger.add is attached automatically.
        es.optimize(objective, maxfun=maxfevals, verb_disp=200)
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

    print("\n=== CMA-ES on HierarchySubsystemProblem6 via es.optimize ===")
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


USE_OPTIMIZE = True  # True 用 main_optimize，False 用 main

if __name__ == "__main__":  # pragma: no cover
    if USE_OPTIMIZE:
        main_optimize()
    else:
        main()
