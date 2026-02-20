from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _parse_optimizer_name(path: Path, problem: str) -> str:
    stem = path.stem  # {problem}_{opt}_runsN
    prefix = f"{problem}_"
    if stem.startswith(prefix):
        rest = stem[len(prefix):]
    else:
        rest = stem
    opt_part = rest.split("_runs")[0]
    return opt_part or stem


def load_curves(input_dir: Path, problem: str) -> List[tuple[str, np.ndarray, Optional[np.ndarray]]]:
    curves = []
    target_prefix = f"{problem.lower()}_"
    for npz_path in sorted(input_dir.glob("*.npz")):
        stem_lower = npz_path.stem.lower()
        if not stem_lower.startswith(target_prefix):
            continue
        data = np.load(npz_path)
        fe = data["fe"]
        mean = data["mean"]
        std = data["std"] if "std" in data else None
        label = _parse_optimizer_name(npz_path, problem)
        curves.append((label, fe, mean, std))
    return curves


def plot_curves(
    curves: List[tuple[str, np.ndarray, Optional[np.ndarray]]],
    out_path: Path,
    logy: bool,
    show: bool,
    max_fe: Optional[int],
) -> None:
    plt.figure(figsize=(8, 5))
    for label, fe, mean, std in curves:
        fe_plot = fe
        mean_plot = mean
        std_plot = std
        if max_fe is not None:
            mask = fe <= max_fe
            if not np.any(mask):
                continue
            fe_plot = fe[mask]
            mean_plot = mean[mask]
            std_plot = std[mask] if std is not None else None

        plt.plot(fe_plot, mean_plot, label=label)
        if std_plot is not None:
            plt.fill_between(fe_plot, mean_plot - std_plot, mean_plot + std_plot, alpha=0.15)

    plt.xlabel("Function evaluations (FEs)")
    plt.ylabel("Best-so-far f")
    # Title includes the problem name (derived from output filename stem)
    plt.title(out_path.stem.replace("_mean_convergence", ""))
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if max_fe is not None:
        plt.xlim(1, max_fe)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    # Configuration (edit these values as needed).
    problem: Optional[str] = "ackley"
    input_dir: str = "convergence"
    out_dir: str = "convergence"
    logy: bool = False
    show: bool = False
    max_fe: Optional[int] = 100000

    input_dir_path = Path(input_dir)
    if not input_dir_path.exists():
        raise SystemExit(f"Input directory not found: {input_dir_path}")

    if problem is None:
        candidates = sorted(input_dir_path.glob("*_runs*.npz"))
        if not candidates:
            raise SystemExit(f"No .npz files found in {input_dir_path}")
        problems = {p.stem.split("_")[0] for p in candidates}
        if len(problems) != 1:
            raise SystemExit(f"Multiple problems found ({', '.join(sorted(problems))}); please provide one explicitly.")
        problem = next(iter(problems))

    curves = load_curves(input_dir_path, problem)
    if not curves:
        raise SystemExit(f"No files found matching {problem}_*_runs*.npz in {input_dir_path}")

    out_path = Path(out_dir) / f"{problem}_mean_convergence.png"
    plot_curves(curves, out_path, logy=logy, show=show, max_fe=max_fe)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
