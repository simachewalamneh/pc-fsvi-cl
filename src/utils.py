"""Utilities: RNG seeding, RBF kernel, synthetic task data generators."""
import numpy as np


def set_seed(seed: int):
    rng = np.random.default_rng(seed)
    return rng


def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """RBF / squared-exponential kernel.
    X1: (n,d), X2: (m,d) -> (n,m)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    if X1.shape[1] != X2.ndim and X1.ndim == 1:
        pass
    sq = (
        np.sum(X1 ** 2, axis=1)[:, None]
        + np.sum(X2 ** 2, axis=1)[None, :]
        - 2 * X1 @ X2.T
    )
    return variance * np.exp(-0.5 * sq / lengthscale ** 2)


def make_task(task_id: int, n=40, x_range=(-3, 3), noise_std=0.1, rng=None):
    """Synthetic 1D regression tasks used throughout the experiments.
    Task 1: y = sin(x) + eps
    Task 2: y = sin(x) + 0.5 + eps
    Task 3: y = cos(x) + eps
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(*x_range, size=n))
    if task_id == 1:
        f = np.sin(x)
    elif task_id == 2:
        f = np.sin(x) + 0.5
    elif task_id == 3:
        f = np.cos(x)
    else:
        raise ValueError(f"Unknown task_id {task_id}")
    y = f + rng.normal(0, noise_std, size=n)
    return x.reshape(-1, 1), y, f


def true_function(task_id: int, x):
    x = np.asarray(x).reshape(-1)
    if task_id == 1:
        return np.sin(x)
    elif task_id == 2:
        return np.sin(x) + 0.5
    elif task_id == 3:
        return np.cos(x)
    raise ValueError(f"Unknown task_id {task_id}")
