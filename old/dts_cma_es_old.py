"""
DTS-CMA-ES: a doubly trained surrogate wrapper around pycma's CMAEvolutionStrategy.

The algorithm follows:
- Z. Pitra et al., PPSN 2016
- L. Bajer et al., ECJ 2019

Only standard ask/tell of CMA-ES is used; pycma itself is not modified.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import LinAlgError

import cma
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


ArrayLike = Sequence[float]
Objective = Callable[[np.ndarray], float]


@dataclass
class AlphaState:
    """Keeps the current alpha (true-eval fraction) and ranking error EMA."""

    alpha: float = 0.05
    err: float = 0.5


@dataclass
class WhiteningTransform:
    """Holds whitening / normalization information."""

    mean: np.ndarray
    L: np.ndarray
    y_mean: float
    y_std: float


@dataclass
class DTSResult:
    """Result container returned by dts_cma_es."""

    best_x: np.ndarray
    best_f: float
    evaluations: int
    iterations: int
    archive_X: List[np.ndarray]
    archive_y: List[float]
    true_evals_per_gen: List[int]
    es_result: Any


def _default_gp_kernel(dim: int):
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dim),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-3))


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))


def _current_covariance(es: cma.CMAEvolutionStrategy, dim: int) -> np.ndarray:
    cov = None
    try:
        cov = es.sm.covariance_matrix
    except Exception:
        cov = None
    if cov is None:
        cov = getattr(es, "C", None)
    cov = np.asarray(cov) if cov is not None else None
    if cov is None or cov.shape != (dim, dim):
        return np.eye(dim)
    return cov


def select_training_data(
    archive_X: Sequence[np.ndarray],
    archive_y: Sequence[float],
    mean: np.ndarray,
    sigma: float,
    C: np.ndarray,
    n_min: int,
    n_max: int,
    r_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick archive points within a Mahalanobis radius, clipped to [n_min, n_max]."""
    if len(archive_X) == 0:
        return np.empty((0, len(mean))), np.empty((0,))
    Sigma = sigma**2 * C
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)

    def dist2(x: np.ndarray) -> float:
        d = x - mean
        return float(d.T @ Sigma_inv @ d)

    d2 = np.array([dist2(np.asarray(x)) for x in archive_X])
    inside = np.where(d2 <= r_max * r_max)[0]
    if len(inside) < n_min:
        inside = np.argsort(d2)[: max(n_min, min(len(d2), n_max))]
    elif len(inside) > n_max:
        inside = inside[np.argsort(d2[inside])[:n_max]]

    X_train = np.asarray([archive_X[i] for i in inside])
    y_train = np.asarray([archive_y[i] for i in inside], dtype=float)
    return X_train, y_train


def whiten_and_normalize(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mean: np.ndarray,
    sigma: float,
    C: np.ndarray,
    jitter: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, WhiteningTransform]:
    """Whiten inputs with Sigma and standardize outputs."""
    dim = len(mean)
    Sigma = sigma**2 * C
    Sigma = (Sigma + Sigma.T) * 0.5
    Sigma += jitter * np.eye(dim)
    try:
        L = np.linalg.cholesky(Sigma)
    except LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.clip(eigvals, jitter, None)
        L = eigvecs @ np.diag(eigvals**0.5)

    centered = X_train - mean
    Xw = np.linalg.solve(L, centered.T).T

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-12:
        y_std = 1.0
    yw = (y_train - y_mean) / y_std

    return Xw, yw, WhiteningTransform(mean=mean, L=L, y_mean=y_mean, y_std=y_std)


def apply_whitening(X: np.ndarray, transform: WhiteningTransform) -> np.ndarray:
    centered = X - transform.mean
    return np.linalg.solve(transform.L, centered.T).T


def train_gp(
    Xw: np.ndarray,
    yw: np.ndarray,
    dim: int,
    gp_params: Optional[Dict[str, Any]] = None,
) -> GaussianProcessRegressor:
    params = dict(gp_params) if gp_params else {}
    kernel = params.pop("kernel", _default_gp_kernel(dim))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=params.pop("alpha", 0.0),
        normalize_y=params.pop("normalize_y", False),
        n_restarts_optimizer=params.pop("n_restarts_optimizer", 1),
        **params,
    )
    gp.fit(Xw, yw)
    return gp


def gp_predict(
    gp: GaussianProcessRegressor,
    X: np.ndarray,
    transform: WhiteningTransform,
) -> Tuple[np.ndarray, np.ndarray]:
    Xw = apply_whitening(X, transform)
    mu_w, std_w = gp.predict(Xw, return_std=True)
    mu = mu_w * transform.y_std + transform.y_mean
    std = std_w * transform.y_std
    return mu, std


def select_points_to_evaluate(
    mu: np.ndarray,
    std: np.ndarray,
    n_orig: int,
    best_y: float,
    criterion: str = "variance",
) -> List[int]:
    n_orig = max(1, min(n_orig, len(mu)))
    if criterion == "poi":
        threshold = best_y - 0.05 * (np.max(mu) - best_y)
        z = (threshold - mu) / (std + 1e-12)
        poi = _norm_cdf(z)
        scores = -poi
    else:  # variance
        scores = -std
    return list(np.argsort(scores)[:n_orig])


def ranking_error(ref: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of pairwise order disagreements between ref and pred."""
    n = len(ref)
    if n < 2:
        return 0.0
    order_ref = np.argsort(ref)
    order_pred = np.argsort(pred)
    pos_pred = np.empty(n, dtype=int)
    pos_pred[order_pred] = np.arange(n)
    perm = pos_pred[order_ref]
    discord = 0
    for i in range(n):
        discord += np.sum(perm[i] > perm[i + 1 :])
    total = n * (n - 1) / 2.0
    return float(discord / total)


def _evaluate_all(
    X: Sequence[np.ndarray],
    objective: Objective,
    archive_X: List[np.ndarray],
    archive_y: List[float],
    known: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, int]:
    y = np.empty(len(X))
    true_evals = 0
    known = known or {}
    for i, x in enumerate(X):
        if i in known:
            y[i] = known[i]
            continue
        y[i] = float(objective(np.asarray(x)))
        # use np.array to ensure a copy without relying on np.asarray(copy=...)
        archive_X.append(np.array(x, copy=True))
        archive_y.append(float(y[i]))
        true_evals += 1
    return y, true_evals


def dts_generation(
    es: cma.CMAEvolutionStrategy,
    X: Sequence[np.ndarray],
    archive_X: List[np.ndarray],
    archive_y: List[float],
    alpha_state: AlphaState,
    objective: Objective,
    gp_params: Optional[Dict[str, Any]],
    dts_params: Dict[str, Any],
) -> Tuple[np.ndarray, List[np.ndarray], List[float], AlphaState, int]:
    dim = len(X[0])
    sigma = float(es.sigma)
    C = _current_covariance(es, dim)
    warmup_min = dts_params["warmup_min"]
    true_evals = 0

    # Warm-up: evaluate everything
    if len(archive_X) < warmup_min:
        y, n_new = _evaluate_all(X, objective, archive_X, archive_y)
        true_evals += n_new
        return y, archive_X, archive_y, alpha_state, true_evals

    # First GP
    X_train, y_train = select_training_data(
        archive_X,
        archive_y,
        mean=np.asarray(es.mean),
        sigma=sigma,
        C=C,
        n_min=dts_params["n_min_train"],
        n_max=dts_params["n_max_train"],
        r_max=dts_params["r_max"],
    )
    if len(X_train) < dts_params["n_min_train"]:
        y, n_new = _evaluate_all(X, objective, archive_X, archive_y)
        true_evals += n_new
        return y, archive_X, archive_y, alpha_state, true_evals

    try:
        Xw, yw, transform = whiten_and_normalize(
            X_train,
            y_train,
            mean=np.asarray(es.mean),
            sigma=sigma,
            C=C,
            jitter=dts_params.get("jitter", 1e-12),
        )
        gp1 = train_gp(Xw, yw, dim, gp_params)
        mu1, std1 = gp_predict(gp1, np.asarray(X), transform)
    except Exception:
        y, n_new = _evaluate_all(X, objective, archive_X, archive_y)
        true_evals += n_new
        return y, archive_X, archive_y, alpha_state, true_evals

    best_true = float(np.min(archive_y))
    n_orig = max(
        dts_params["min_true_per_gen"],
        int(math.ceil(alpha_state.alpha * len(X))),
    )
    idx_eval = select_points_to_evaluate(
        mu=mu1,
        std=std1,
        n_orig=n_orig,
        best_y=best_true,
        criterion=dts_params["criterion"],
    )

    evaluated: Dict[int, float] = {}
    for idx in idx_eval:
        fx = float(objective(np.asarray(X[idx])))
        evaluated[idx] = fx
        archive_X.append(np.array(X[idx], copy=True))
        archive_y.append(fx)
        true_evals += 1

    # Second GP on updated archive
    X_train2, y_train2 = select_training_data(
        archive_X,
        archive_y,
        mean=np.asarray(es.mean),
        sigma=sigma,
        C=C,
        n_min=dts_params["n_min_train"],
        n_max=dts_params["n_max_train"],
        r_max=dts_params["r_max"],
    )
    if len(X_train2) < dts_params["n_min_train"]:
        y, n_new = _evaluate_all(X, objective, archive_X, archive_y, known=evaluated)
        true_evals += n_new
        return y, archive_X, archive_y, alpha_state, true_evals

    try:
        Xw2, yw2, transform2 = whiten_and_normalize(
            X_train2,
            y_train2,
            mean=np.asarray(es.mean),
            sigma=sigma,
            C=C,
            jitter=dts_params.get("jitter", 1e-12),
        )
        gp2 = train_gp(Xw2, yw2, dim, gp_params)
        mu2, std2 = gp_predict(gp2, np.asarray(X), transform2)
    except Exception:
        y, n_new = _evaluate_all(X, objective, archive_X, archive_y, known=evaluated)
        true_evals += n_new
        return y, archive_X, archive_y, alpha_state, true_evals

    y_sur = mu2.copy()
    for idx, val in evaluated.items():
        y_sur[idx] = val

    clamp = dts_params.get("clamp_best", None)
    if clamp is not None:
        y_sur = np.maximum(y_sur, best_true - float(clamp))

    # Adapt alpha from ranking error
    err = ranking_error(mu1, y_sur)
    beta = dts_params["beta"]
    alpha_state.err = (1.0 - beta) * alpha_state.err + beta * err
    e = float(np.clip(alpha_state.err, dts_params["e_min"], dts_params["e_max"]))
    if dts_params["e_max"] > dts_params["e_min"]:
        alpha_state.alpha = dts_params["alpha_min"] + (
            dts_params["alpha_max"] - dts_params["alpha_min"]
        ) * (e - dts_params["e_min"]) / (
            dts_params["e_max"] - dts_params["e_min"]
        )
    else:
        alpha_state.alpha = dts_params["alpha_max"]
    alpha_state.alpha = float(np.clip(alpha_state.alpha, dts_params["alpha_min"], dts_params["alpha_max"]))

    return y_sur, archive_X, archive_y, alpha_state, true_evals


def _default_dts_params(dim: int) -> Dict[str, Any]:
    return {
        "alpha_min": 0.04,
        "alpha_max": 1.0,
        "alpha_init": 0.05,
        "beta": 0.2,
        "e_min": 0.0,
        "e_max": 0.5,
        "warmup_min": max(10 * dim, 50),
        "n_min_train": 10 * dim,
        "n_max_train": 20 * dim,
        "r_max": 4.0 * math.sqrt(max(dim, 1)),
        "criterion": "variance",
        "min_true_per_gen": 1,
        "jitter": 1e-12,
        "clamp_best": None,
    }


def dts_cma_es(
    objective: Objective,
    x0: ArrayLike,
    sigma0: float,
    max_evals: int,
    gp_params: Optional[Dict[str, Any]] = None,
    dts_params: Optional[Dict[str, Any]] = None,
    cma_options: Optional[Dict[str, Any]] = None,
) -> DTSResult:
    """
    Run DTS-CMA-ES on `objective` starting at `x0` with initial sigma `sigma0`.

    Returns a DTSResult with pycma's result included as `es_result`.
    """
    x0 = np.asarray(x0, dtype=float)
    dim = len(x0)
    params = _default_dts_params(dim)
    if dts_params:
        params.update(dts_params)
    cma_opts = cma_options or {}

    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)
    archive_X: List[np.ndarray] = []
    archive_y: List[float] = []
    alpha_state = AlphaState(alpha=params["alpha_init"], err=params["e_max"])
    true_evals_per_gen: List[int] = []
    eval_count = 0

    while not es.stop() and eval_count < max_evals:
        X = es.ask()
        fitnesses, archive_X, archive_y, alpha_state, n_true = dts_generation(
            es=es,
            X=X,
            archive_X=archive_X,
            archive_y=archive_y,
            alpha_state=alpha_state,
            objective=objective,
            gp_params=gp_params,
            dts_params=params,
        )
        eval_count += n_true
        true_evals_per_gen.append(n_true)
        # Print best true fitness seen so far each generation.
        try:
            disp_gap = int(es.opts["verb_disp"])
        except Exception:
            disp_gap = 0
        if archive_y and disp_gap > 0 and len(true_evals_per_gen) % disp_gap == 0:
            best_so_far = min(archive_y)
            print(f"[DTS] iter {len(true_evals_per_gen):4d} | best f = {best_so_far:.6e}")
        es.tell(X, fitnesses)

    es_res = es.result
    return DTSResult(
        best_x=np.asarray(es_res.xbest),
        best_f=float(es_res.fbest),
        evaluations=int(es_res.evaluations),
        iterations=int(es_res.iterations),
        archive_X=archive_X,
        archive_y=archive_y,
        true_evals_per_gen=true_evals_per_gen,
        es_result=es_res,
    )


if __name__ == "__main__":

    import warnings
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # # Example 1: 100-D Sphere (simple sanity check)
    # def sphere(x):
    #     return float(np.sum(np.array(x) ** 2))

    # res_sphere = dts_cma_es(
    #     objective=sphere,
    #     x0=np.zeros(10),
    #     sigma0=0.5,
    #     max_evals=5000,
    #     gp_params=None,
    #     dts_params={"criterion": "variance"},
    #     cma_options={"seed": 0, "verb_disp": 10},
    # )
    # print("=== DTS-CMA-ES on 100-D Sphere ===")
    # print(f"best f: {res_sphere.best_f:.6e}")
    # print(f"best x (first 5 dims): {res_sphere.best_x[:5]}")
    # print(f"true evals used: {sum(res_sphere.true_evals_per_gen)}")

    # Example 1b: Ackley (dim-scalable)
    def ackley(x):
        x = np.asarray(x, dtype=float)
        d = x.size
        a, b, c = 20.0, 0.2, 2 * math.pi
        sum_sq = np.sum(x * x)
        sum_cos = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * math.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return float(term1 + term2 + a + math.e)

    ack_dim = 20
    rng = np.random.RandomState(42)
    res_ack = dts_cma_es(
        objective=ackley,
        # x0=np.zeros(ack_dim),
        x0=rng.uniform(-32.0, 32.0, size=ack_dim),
        sigma0=3.0,
        max_evals=3000,
        gp_params=None,
        dts_params={"criterion": "variance"},
        cma_options={"seed": 1, "verb_disp": 10},
    )
    print("=== DTS-CMA-ES on Ackley ===")
    print(f"best f: {res_ack.best_f:.6e}")
    print(f"best x (first 5 dims): {res_ack.best_x[:5]}")
    print(f"true evals used: {sum(res_ack.true_evals_per_gen)}")

    # Example 2: 100-D HierarchicalSubsystem benchmark.
    # from problems.hierarchical_subsystem import HierarchicalSubsystem

    # problem = HierarchicalSubsystem()
    # dim = problem.dim

    # def objective(x):
    #     return float(problem(np.asarray(x)))

    # x0 = np.zeros(dim)
    # sigma0 = (problem.upper_bound - problem.lower_bound) / 10.0  # moderate initial step size

    # res = dts_cma_es(
    #     objective=objective,
    #     x0=x0,
    #     sigma0=sigma0,
    #     max_evals=10000,  # budget counts true evaluations only
    #     gp_params=None,
    #     dts_params={"criterion": "variance"},
    #     cma_options={
    #         "seed": 0,
    #         "bounds": [problem.lower_bound, problem.upper_bound],
    #         "verb_disp": 5,
    #     },
    # )

    # print("=== DTS-CMA-ES on HierarchicalSubsystem (100-D) ===")
    # print(f"best f: {res.best_f:.6e}")
    # print(f"best x (first 5 dims): {res.best_x[:5]}")
    # print(f"true evals used: {sum(res.true_evals_per_gen)}")
