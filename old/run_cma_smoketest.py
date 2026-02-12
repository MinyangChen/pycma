"""Quick smoke test for the local ``pycma`` checkout.

Usage:
- make sure you are in the repository root next to the ``cma`` package
- run: ``python run_cma_smoketest.py`` (optionally set ``CMA_SEED=<int>`` for repeatability)

The script minimizes a 10-D Rosenbrock function with CMA-ES and prints the best
solution found. If you want to use the library in your own code, just copy the
``cma.CMAEvolutionStrategy`` loop below and replace the objective function.
"""
from __future__ import annotations

import os
import time
from typing import Sequence

import cma  # uses the local package in this repo when run from the root


def rosenbrock(x: Sequence[float]) -> float:
    """Classic non-convex test function with minimum at x_i = 1."""
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1))


def main() -> None:
    if not hasattr(cma, "CMAEvolutionStrategy"):
        raise SystemExit(
            "The full CMA-ES implementation is unavailable (likely missing numpy). "
            "Install with `pip install -e .` from the repo root or `pip install numpy` first."
        )

    dim = 10
    x0 = [1.2] * dim  # start a bit away from the optimum at all-ones
    sigma0 = 0.3
    seed = int(os.environ["CMA_SEED"]) if "CMA_SEED" in os.environ else None

    es = cma.CMAEvolutionStrategy(x0, sigma0, {"seed": seed, "maxiter": 200, "verb_disp": 25})

    t0 = time.time()
    while not es.stop():
        candidates = es.ask()
        es.tell(candidates, [rosenbrock(x) for x in candidates])
        es.logger.add()  # keeps a log you can plot later via cma.plot()

    duration = time.time() - t0
    es.result_pretty()

    best_x, best_fx = es.result.xbest, es.result.fbest
    print(f"\nFinished in {duration:.2f}s, best f(x) = {best_fx:.3e}")
    if abs(best_fx) < 1e-6:
        print("Success: reached the Rosenbrock minimum near zero.")
    else:
        print("Finished: best value above tight tolerance; tweak maxiter/sigma0 if needed.")


if __name__ == "__main__":  # pragma: no cover
    main()
