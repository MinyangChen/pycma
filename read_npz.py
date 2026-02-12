from __future__ import annotations

from pathlib import Path

import numpy as np

# List the .npz files you want to inspect here.
# Example entries below cover the current convergence outputs; edit as needed.
FILES = [
    # "convergence/AdditiveSubsystem_cma_runs10.npz",
    # "convergence/AdditiveSubsystem_lshade_runs10.npz",
    "convergence/AdditiveSubsystem_ms_dts_cma_runs1.npz",
    # "convergence/AdditiveSubsystem_ms_lshade_runs10.npz",
    # "convergence/AdditiveSubsystem_ms_shade_runs10.npz",
    # "convergence/AdditiveSubsystem_shade_runs10.npz",
    # "convergence/HierarchicalSubsystem_cma_runs10.npz",
    # "convergence/HierarchicalSubsystem_lshade_runs10.npz",
    "convergence/HierarchicalSubsystem_ms_dts_cma_runs1.npz",
    # "convergence/HierarchicalSubsystem_ms_lshade_runs10.npz",
    # "convergence/HierarchicalSubsystem_ms_shade_runs10.npz",
    # "convergence/HierarchicalSubsystem_shade_runs10.npz",
]


def describe_npz(path: Path) -> None:
    data = np.load(path)
    keys = list(data.keys())
    print(f"File: {path.name}")
    print(f"  keys: {keys}")
    for k in ["fe", "curves", "mean", "std", "seeds"]:
        if k in data:
            arr = data[k]
            print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
    print("")


def main() -> None:
    paths = [Path(p) for p in FILES]
    if not paths:
        raise SystemExit("No files listed in FILES.")

    found_any = False
    for p in paths:
        if not p.exists():
            print(f"Skipping (not found): {p}")
            continue
        found_any = True
        describe_npz(p)

    if not found_any:
        raise SystemExit("No files were found. Update FILES with existing .npz paths.")


if __name__ == "__main__":
    main()
