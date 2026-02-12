"""Smoke test for the local ``pycma`` checkout using a max-evaluations stop.

Run from the repo root (uses the local ``cma`` package):
  python run_cma_smoketest_fevals.py
Optionally set CMA_SEED for repeatability:
  CMA_SEED=0 python run_cma_smoketest_fevals.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Sequence

import cma


def main() -> None:
    if not hasattr(cma, "CMAEvolutionStrategy"):
        sys.exit(
            "The full CMA-ES implementation is unavailable (likely missing numpy). "
            "Install with `pip install -e .` from the repo root or `pip install numpy` first."
        )

    # Use the built-in Rosenbrock test function from this project.
    objective = cma.ff.rosen  # type: ignore[attr-defined]
    dim = 10
    x0: Sequence[float] = [0.0] * dim
    sigma0 = 0.5
    seed = int(os.environ["CMA_SEED"]) if "CMA_SEED" in os.environ else None

    # Stop criterion uses function evaluations instead of iterations.
    options = {
        "seed": seed,
        "maxfevals": 8000,  # adjust as needed
        "verb_disp": 50,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, options)

    t0 = time.time()
    while not es.stop():
        candidates = es.ask()
        es.tell(candidates, [objective(x) for x in candidates])
        es.logger.add()

    duration = time.time() - t0
    es.result_pretty()

    best_x, best_fx = es.result.xbest, es.result.fbest
    print(f"\nFinished in {duration:.2f}s, best f(x) = {best_fx:.3e}")
    if "maxfevals" in es.stop() and best_fx > 1e-6:
        print("Stopped on maxfevals; increase the budget for higher accuracy.")
    elif abs(best_fx) < 1e-6:
        print("Success: reached a tight tolerance near the Rosenbrock minimum.")


if __name__ == "__main__":  # pragma: no cover
    main()
