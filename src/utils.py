"""Utilities: RNG seeding, RBF kernel, synthetic task data generators."""
import numpy as np


def set_seed(seed: int):
    return np.random.default_rng(seed)


def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """RBF / squared-exponential kernel. X1: (n,d), X2: (m,d) -> (n,m)."""
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sq = (
        np.sum(X1 ** 2, axis=1)[:, None]
        + np.sum(X2 ** 2, axis=1)[None, :]
        - 2 * X1 @ X2.T
    )
    return variance * np.exp(-0.5 * sq / lengthscale ** 2)


def true_function(task_id: int, x):
    x = np.asarray(x).reshape(-1)
    if task_id == 1:
        return np.sin(x)
    elif task_id == 2:
        return np.sin(x) + 0.5
    elif task_id == 3:
        return np.cos(x)
    raise ValueError(f"Unknown task_id {task_id}")


def make_task(task_id: int, n=40, x_range=(-3, 3), noise_std=0.1, rng=None):
    """CONFLICTING benchmark: same inputs, different functions.
    Task 1: sin(x); Task 2: sin(x)+0.5; Task 3: cos(x)  (+ noise)
    A single-function model cannot fit all three, so 'forgetting' here is
    model misspecification, not a property of the CL method.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(*x_range, size=n))
    f = true_function(task_id, x)
    y = f + rng.normal(0, noise_std, size=n)
    return x.reshape(-1, 1), y, f


# ---- NON-CONFLICTING benchmark: one function, tasks = input regions -------
REGIONS = {1: (-3.0, -1.0), 2: (-1.0, 1.0), 3: (1.0, 3.0)}


def make_regional_task(task_id: int, n=30, noise_std=0.1, rng=None):
    """y = sin(x) + eps, with x restricted to REGIONS[task_id]."""
    if rng is None:
        rng = np.random.default_rng(0)
    lo, hi = REGIONS[task_id]
    x = np.sort(rng.uniform(lo, hi, size=n))
    f = np.sin(x)
    y = f + rng.normal(0, noise_std, size=n)
    return x.reshape(-1, 1), y, f


def make_benchmark(name: str, seed=0, n=30, noise_std=0.1, n_test=100):
    """Return (task_data, eval_data).

    task_data : list of (X, y, f_true) training sets, one per task
    eval_data : list of (X_test, f_test) on a dense HELD-OUT grid per task
                (the task's own input range, noise-free targets)
    name      : "conflicting" or "regional"
    """
    rng = np.random.default_rng(seed)
    if name == "conflicting":
        task_data = [make_task(t, n=n, noise_std=noise_std, rng=rng) for t in (1, 2, 3)]
        grid = np.linspace(-3, 3, n_test).reshape(-1, 1)
        eval_data = [(grid, true_function(t, grid)) for t in (1, 2, 3)]
    elif name == "regional":
        task_data = [make_regional_task(t, n=n, noise_std=noise_std, rng=rng) for t in (1, 2, 3)]
        eval_data = []
        for t in (1, 2, 3):
            lo, hi = REGIONS[t]
            g = np.linspace(lo, hi, n_test).reshape(-1, 1)
            eval_data.append((g, np.sin(g).ravel()))
    else:
        raise ValueError(name)
    return task_data, eval_data