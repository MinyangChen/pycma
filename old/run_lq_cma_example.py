#!/usr/bin/env python
"""Quick demo of surrogate-assisted lq-CMA-ES on Rosenbrock.

Run from the repo root (uses the local `cma` package):
    python run_lq_cma_example.py
Set CMA_SEED for reproducibility:
    CMA_SEED=0 python run_lq_cma_example.py
"""

import os
import sys
import time

try:
    import numpy  # noqa: F401  # required for full `cma`
except ImportError as exc:
    sys.exit("numpy is required for lq-CMA-ES. Install with `python -m pip install numpy`: %s" % exc)

import cma


def main() -> None:
    if not hasattr(cma, "fmin_lq_surr2"):
        sys.exit("cma.fmin_lq_surr2 not found; ensure you're running this repo's `cma` package.")

    dim = 5
    x0 = [0.3] * dim
    sigma0 = 0.4
    seed = int(os.environ["CMA_SEED"]) if "CMA_SEED" in os.environ else 0

    opts = {
        "seed": seed,
        # Stop by number of function evaluations rather than iterations.
        "maxfevals": 5000,  # adjust as needed
        "ftarget": 1e-6,
        "bounds": [-2, 2],  # same bounds for all dimensions
        "verb_disp": 20,
    }

    t0 = time.time()
    # incpopsize is a dedicated fmin_lq_surr2 argument, not a CMA option
    xbest, es = cma.fmin_lq_surr2(cma.ff.rosen, x0, sigma0, opts, incpopsize=2)
    elapsed = time.time() - t0

    print("\n=== lq-CMA-ES result ===")
    print(es.result_pretty())
    print(f"xfavorite (best estimate from true evaluations): {es.result.xfavorite}")
    print(f"xbest returned (may be surrogate-best): {xbest}")
    print(f"iterations: {es.countiter}, evaluations: {es.result.evaluations}")
    print(f"elapsed: {elapsed:.2f}s")


if __name__ == "__main__":  # pragma: no cover
    main()
