from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type

import numpy as np


@dataclass
class RunResult:
    seed: int
    best_x: np.ndarray
    best_f: float
    history: List[float]
    n_iters: Optional[int] = None


class ExperimentRunner:
    """Benchmark multiple optimizers across problems with a unified interface."""

    def __init__(
        self,
        optimizers: Mapping[str, Type],
        problems: Mapping[str, Type],
    ) -> None:
        self.optimizers = dict(optimizers)
        self.problems = dict(problems)

    def run(
        self,
        max_evals: int,
        seeds: Iterable[int],
        optimizer_configs: Mapping[str, Dict[str, Any]],
        problem_configs: Mapping[str, Dict[str, Any]],
    ) -> Dict[Tuple[str, str], List[RunResult]]:
        results: Dict[Tuple[str, str], List[RunResult]] = {}

        for prob_name, prob_cls in self.problems.items():
            prob_cfg = dict(problem_configs.get(prob_name, {}))
            problem = prob_cls(**prob_cfg)
            dim = getattr(problem, "dim", None)
            if dim is None:
                raise ValueError(f"Problem '{prob_name}' must expose a 'dim' attribute.")

            lower = np.asarray(getattr(problem, "lower_bound", -np.inf), dtype=float)
            upper = np.asarray(getattr(problem, "upper_bound", np.inf), dtype=float)
            if lower.size == 1:
                lower = np.full(dim, float(lower))
            if upper.size == 1:
                upper = np.full(dim, float(upper))
            span = 0.5 * (upper - lower)
            if span.size == 0 or not np.isfinite(span).all():
                sigma0_auto = 0.5
            else:
                sigma0_auto = 0.30 * float(np.median(span))

            for opt_name, opt_cls in self.optimizers.items():
                base_cfg = optimizer_configs.get(opt_name, {})
                opt_cfg = dict(base_cfg) if base_cfg is not None else {}
                # Only fill optional parameters when the config declares them.
                if "sigma0" in opt_cfg and opt_cfg["sigma0"] is None:
                    opt_cfg["sigma0"] = sigma0_auto
                if "bounds" in opt_cfg and opt_cfg["bounds"] is None:
                    opt_cfg["bounds"] = (lower, upper)

                key = (prob_name, opt_name)
                results[key] = []

                for seed in seeds:
                    optimizer = opt_cls(dim=dim, seed=seed, **opt_cfg)
                    best_x, best_f, history = optimizer.optimize(problem, max_evals)
                    n_iters = len(history) if history else None
                    results[key].append(
                        RunResult(
                            seed=seed,
                            best_x=np.asarray(best_x, dtype=float),
                            best_f=float(best_f),
                            history=list(history),
                            n_iters=n_iters,
                        )
                    )

        self._print_summary(results)
        return results

    @staticmethod
    def _print_summary(results: Dict[Tuple[str, str], List[RunResult]]) -> None:
        for (prob_name, opt_name), runs in results.items():
            best_values = np.array([r.best_f for r in runs], dtype=float)
            mean = float(np.mean(best_values)) if best_values.size else float("nan")
            std = float(np.std(best_values)) if best_values.size else float("nan")
            best_list = ", ".join(f"{v:.4g}" for v in best_values)
            print(f"[{prob_name} | {opt_name}] best_f per seed: [{best_list}]  (mean={mean:.4g}, std={std:.4g})")


def _history_to_curve(history: List[float], max_evals: int) -> np.ndarray:
    """
    Convert per-iteration history into a length-max_evals FE curve with step-hold.
    pos[t] = round(1 + t*(max_evals-1)/(T-1)), with prefix min.
    """
    if max_evals <= 0:
        return np.array([], dtype=float)

    if not history:
        return np.full(max_evals, np.inf, dtype=float)

    hist = np.minimum.accumulate(np.asarray(history, dtype=float))
    T = hist.size

    if T == 1:
        pos = np.array([1], dtype=int)
    else:
        t = np.arange(T, dtype=float)
        pos = np.round(1.0 + t * (max_evals - 1) / (T - 1)).astype(int)

    pos = np.clip(pos, 1, max_evals)

    curve = np.full(max_evals, hist[-1], dtype=float)
    last_idx = 0
    for val, p in zip(hist, pos):
        idx = max(int(p) - 1, last_idx)
        idx = min(idx, max_evals - 1)
        curve[last_idx: idx + 1] = float(val)
        last_idx = idx + 1
        if last_idx >= max_evals:
            break

    if last_idx < max_evals:
        curve[last_idx:] = hist[-1]

    return curve


def export_convergence(
    results: Dict[Tuple[str, str], List[RunResult]],
    max_evals: int,
    out_dir: str = "convergence",
) -> None:
    """
    Export per-run FE curves plus mean/std for each (problem, optimizer).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fe = np.arange(1, max_evals + 1, dtype=int)

    for (prob_name, opt_name), runs in results.items():
        if not runs:
            continue
        curves = np.vstack([_history_to_curve(r.history, max_evals) for r in runs])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        seeds = np.array([r.seed for r in runs], dtype=int)

        fname = f"{prob_name}_{opt_name}_runs{len(runs)}.npz"
        np.savez(out_path / fname, fe=fe, curves=curves, mean=mean, std=std, seeds=seeds)


def main() -> None:
    """Run a small demo benchmark when executed as a script."""
    from optimizers.pycma_cmaes import PyCMAESOptimizer
    from optimizers.pycma_lqcmaes import PyLQCMAESOptimizer
    from optimizers.dts_cmaes import DTSCMAESOptimizer
    from optimizers.ms_dts_cmaes import MSDTSCMAESOptimizer
    from optimizers.shade import SHADEOptimizer
    from optimizers.lshade import LSHADEOptimizer
    from optimizers.ms_shade import MSSHADEOptimizer
    from optimizers.shade_gp import SHADEGPOptimizer
    from optimizers.gp_shade import GPSHADEOptimizer
    from optimizers.de_gp import DEGPOptimizer
    from optimizers.ms_lshade import MSLSHADEOptimizer
    from optimizers.dts_brq_cmaes import DTSBRQ_CMAESOptimizer
    from problems.ackley import AckleyProblem
    from problems.additive_subsystem_hs5 import AdditiveSubsystem
    from problems.hierarchical_subsystem import HierarchicalSubsystem
    from optimizers.dts_rank_cmaes import DTS_RANK_CMAESOptimizer
    from optimizers.dts_bnn_cmaes import DTSBNN_CMAESOptimizer


    # Use fixed seeds for repeatable runs across optimizers/problems.
    run_times = 3  # number of independent runs per optimizer/problem
    seeds = list(range(1, run_times + 1))

    # seeds = [42]
    max_evals = 10000

    runner = ExperimentRunner(
        optimizers={
            # CMA-ES baseline
            # "cma": PyCMAESOptimizer
            "dts_cma": DTSCMAESOptimizer,
            # "dts_bnn_cma": DTSBNN_CMAESOptimizer,
            # "shade_gp": SHADEGPOptimizer,
            # "gp_shade": GPSHADEOptimizer,
            # "de_gp": DEGPOptimizer,
            # "dts_brq_cma": DTSBRQ_CMAESOptimizer,
            # "ms_dts_cma": MSDTSCMAESOptimizer,
            # "dts_rank_cma": DTS_RANK_CMAESOptimizer,
            # # DE family
            # "shade": SHADEOptimizer,
            # "lshade": LSHADEOptimizer,
            # "ms_shade": MSSHADEOptimizer,
            # "ms_lshade": MSLSHADEOptimizer,
            # "lq_cma": PyLQCMAESOptimizer,
            # "smas_shade": MSSHADEOptimizer,
            # "smas_lshade": MSLSHADEOptimizer,
        },
        problems={
            # Standard benchmark
            "ackley": AckleyProblem,
            # "ackley20": AckleyProblem,
            # "ackley10": AckleyProblem,

            # HS5 variants
            # "AdditiveSubsystem": AdditiveSubsystem,
            # "HierarchicalSubsystem": HierarchicalSubsystem,
        },
    )

    optimizer_configs = {
        "cma": {
            # leave sigma0 and bounds empty; runner will fill per problem
            "sigma0": None,
            "bounds": None,
            "popsize": None,
            "popsize_mode":"double", # default / double
            "verb_disp":10,
        },
        "dts_cma": {
            "sigma0": 20,
            "bounds": None,
            "popsize_mode":"double", # default / double
            "gp_params": None,
            "dts_params": None,
            "options": {},
        },
        "ms_dts_cma": {
            "sigma0": 20,
            "bounds": None,
            "popsize_mode":"double", # default / double
            "options": {},
            "mimic_top_k": 5,
            "mimic_top_pick": 2,
            "mimic_swap_prob": 0.5,
            "inject_true_into_tell": False,
            # "cma_options":{"tolfunhist": 0.0, "tolfun": 0.0}
            "cma_options": {
                "tolfunhist": -1.0,        # 负数基本等价于“别用它停”
                "tolfun": -1.0,
                "tolstagnation": np.inf,   # 或者给个超大整数，比如 10**12
                "tolxstagnation": False,   # 文档明确说 False/负数可关这个
                "tolx": -1.0,
            }
        },
        "dts_brq_cma": {
            "sigma0": 20,
            "bounds": None,
            "popsize_mode":"double",
            # BRQ surrogate settings (optional; these are the defaults)
            "br_candidate_pool": 2500,
            "br_y_std_min": 1e-12,
            "br_fit_intercept": False,
            # Keep unused config keys for runner compatibility (will be ignored)
            "gp_params": None,
            "dts_params": None,
            "options": {},
        },
        # "dts_rank_cma": {
        #     "sigma0": None,
        #     "bounds": None,
        #     "popsize_mode": "double",
        #     # training set strategy
        #     "n_max_train": 1200,
        #     "global_train_frac": 0.10,
        #     "global_best_k": 50,
        #     # rank surrogate params
        #     "rank_ensemble_size": 5,
        #     "rank_hidden_sizes": (256, 256, 128),
        #     "rank_epochs": 60,
        #     "rank_batch_size": 256,
        #     "rank_pairs_per_batch": 2048,
        #     "rank_focus_top_frac": 0.30,
        #     "rank_w_pair": 1.0,
        #     "rank_w_list": 0.5,
        #     "rank_w_reg": 0.10,
        #     "rank_device": "auto",  # uses CUDA if available
        #     "rank_use_amp": True,
        #     "rank_bootstrap": True,
        #     # keep unused keys for compatibility
        #     "gp_params": None,
        #     "dts_params": None,
        #     "options": {},
        # },
        "dts_rank_cma": {
            "sigma0": None,
            "bounds": None,
            "popsize_mode": "default",   # keep lambda small on laptop

            # --- DTS control (most important) ---
            "warmup_min_points": 400,    # was 1000; 20D doesn’t need that much warmup
            # "use_adaptive_alpha": True,
            # "alpha0": 0.35,              # start higher than 0.10
            # "alpha_min": 0.25,           # prevents n_orig collapsing to ~2
            # "alpha_max": 0.80,
            "min_true_per_gen": 6,       # critical: makes ranking diagnostics meaningful
            "prediction_guard": "shift",

            # --- local training set size ---
            "n_max_train": 600,
            "global_train_frac": 0.05,
            "global_best_k": 30,

            # --- surrogate: CPU-friendly ---
            "rank_device": "cpu",
            "rank_use_amp": False,       # AMP is for GPU
            "rank_ensemble_size": 3,     # 5 -> 3
            "rank_hidden_sizes": (128, 128),
            "rank_dropout": 0.05,
            "rank_epochs": 20,           # 60 -> 20
            "rank_batch_size": 128,
            "rank_pairs_per_batch": 512, # 2048 -> 512
            "rank_focus_top_frac": 0.40, # focus more on best region
            "rank_w_pair": 1.0,
            "rank_w_list": 0.2,          # listwise helps but keep light on CPU
            "rank_w_reg": 0.05,
            "rank_patience": 6,

            # keep unused keys for runner compatibility (ignored)
            "gp_params": None,
            "dts_params": None,
            "options": {},
        },
        "dts_bnn_cma": {
            "sigma0": 20,
            "bounds": None,
            "popsize_mode": "double",

            # DTS knobs (keep flow identical, just tune sizes)
            "warmup_min_points": 1000,   # optional: 500 also works, but 100D benefits from more
            "n_max_train": 1000,         # requested 800–1200 range
            "selection_criterion": "cstd",
            "prediction_guard": "shift",
            "alpha0": 0.10,
            "use_adaptive_alpha": False, # can enable if desired

            # BNN knobs
            "bnn_device": "auto",        # uses CUDA if available
            "bnn_use_amp": True,         # enable AMP on A100
            "bnn_mc_samples": 40,        # 20–50 recommended; 40 is a solid default
            "bnn_epochs": 300,           # keep per-gen cost bounded; early stop is on
            "bnn_lr": 2e-2,
            "bnn_lr_gamma": 0.999,
            "bnn_kl_weight": 0.05,       # template used 0.1; 0.05 often works well in practice
            "bnn_prior_sigma": 0.1,

            "print_every": 50,

            # keep unused keys for runner compatibility (ignored by DTSBNN)
            "gp_params": None,
            "dts_params": None,
            "options": {},
        },
        "shade": {
            # bounds/sigma0 are not used in SHADE, but kept for runner compatibility
            "sigma0": None,
            "bounds": None,
            "pop_size": 100,
            "memory_size": None,
            "p_max": 0.2,
            "p_min": None,
        },
        "shade_gp": {
            "pop_size": 100,
            "memory_size": None,
            "p_max": 0.2,
            "p_min": None,
            "warmup_min_points": 200,
            "gp_rank_mode": "lcb",
            "lcb_kappa": 1.0,
            "gp_max_train_size": 300,
            "gp_keep_best": 60,
            "gp_nu": 2.5,
            "gp_n_restarts_optimizer": 0,
            "gp_random_state": None,
            "gp_y_std_min": 1e-12,
        },
        "gp_shade": {
            "pop_size": 100,
            "memory_size": None,
            "p_max": 0.2,
            "p_min": None,
            "warmup_min_points": 200,
            "top_k_true": 10,
            "gp_rank_mode": "lcb",
            "lcb_kappa": 1.0,
            "gp_max_train_size": 300,
            "gp_keep_best": 60,
            "gp_nu": 2.5,
            "gp_n_restarts_optimizer": 0,
            "gp_random_state": None,
            "gp_y_std_min": 1e-12,
            "print_every": 500,
        },
        "de_gp": {
            "pop_size": 100,   # 5 * D, with D=20 for ackley20
            "alpha_size": 1000, # 20 * D
            "train_size": 1000, # 20 * D
            # "F": 0.5,
            "F": 0.5,
            "CR": 0.9,
            "k_true": 1,
            # "strategy": "current-to-best",  # "rand" / "best" / "current-to-best"
            "strategy": "rand",  # "rand" / "best" / "current-to-best"
            "rank_mode": "lcb",  # "lcb" / "mean"
            "kappa": 1.0,
            "gp_nu": 2.5,
            "gp_n_restarts_optimizer": 0,
            "gp_random_state": None,
            "gp_y_std_min": 1e-12,
            "print_every": 10,
        },
        "lshade": {
            "pop_size": 100,
            "memory_size": None,
            "p_max": 0.2,
            "p_min": None,
            "min_pop_size": 4,
        },
        "ms_shade": {
            "pop_size": 100,
            "memory_size": None,
            "p_max": 0.2,
            "p_min": None,
            # Mimic-surrogate ranker settings
            "top_k": 5,
            "top_pick": 2,
            "swap_prob": 0.2,
        },
        "ms_lshade": {
            "pop_size": 100,
            "memory_size": None,
            "p_max": 0.2,
            "p_min": None,
            "min_pop_size": 4,
            # Mimic-surrogate ranker settings
            "top_k": 5,
            "top_pick": 2,
            "swap_prob": 0.2,
        },
        "lq_cma": {
            "sigma0": None,
            "bounds": None,
            "restarts": 0,
            "incpopsize": 2,
            "inject": True,
            "keep_model": False,
        },
        "smas_shade": {
            "pop_size": 100,
            "memory_size": None,
            "p_max": 0.2,
            "p_min": None,
            # Mimic-surrogate ranker settings
            "top_k": 1,
            "top_pick": 1,
            "swap_prob": 0,
        },
        "smas_lshade": {
            "pop_size": 100,
            "memory_size": None,
            "p_max": 0.2,
            "p_min": None,
            "min_pop_size": 4,
            # Mimic-surrogate ranker settings
            "top_k": 1,
            "top_pick": 1,
            "swap_prob": 0,
        },
    }

    problem_configs = {
        "ackley": {"dim": 100, "name": "ackley"},
        "ackley20": {"dim": 20, "name": "ackley"},
        "ackley10": {"dim": 10, "name": "ackley"},
        "AdditiveSubsystem": {},
        "HierarchicalSubsystem": {},
    }

    results = runner.run(
        max_evals=max_evals,
        seeds=seeds,
        optimizer_configs=optimizer_configs,
        problem_configs=problem_configs,
    )

    export_convergence(results, max_evals=max_evals, out_dir="convergence")


if __name__ == "__main__":
    main()
