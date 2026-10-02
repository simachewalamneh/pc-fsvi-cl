"""Predictive, uncertainty, and continual-learning metrics."""
import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def gaussian_nll(y_true, mean, var):
    y_true = np.asarray(y_true)
    return float(
        np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y_true - mean) ** 2 / var)
    )


def calibration_error(y_true, mean, var, n_bins=10):
    from scipy.stats import norm

    y_true = np.asarray(y_true)
    std = np.sqrt(var)
    probs = np.linspace(0.1, 0.9, n_bins)
    errs = []
    for p in probs:
        z = norm.ppf(0.5 + p / 2)
        lower, upper = mean - z * std, mean + z * std
        empirical = np.mean((y_true >= lower) & (y_true <= upper))
        errs.append(abs(empirical - p))
    return float(np.mean(errs))


def accuracy_matrix_to_metrics(R):
    """R[i, j] = RMSE on task j after training on task i (lower is better).

    average_performance : mean final RMSE over all tasks, R[T-1, :]
    forgetting          : mean over j<T-1 of  R[T-1, j] - min_{i in [j, T-2]} R[i, j]
                          (error increase from the best earlier point; >= 0)
    backward_transfer   : mean over j<T-1 of  R[j, j] - R[T-1, j]
                          (conventional sign: NEGATIVE = old tasks got worse,
                           POSITIVE = learning later tasks helped old ones)
    plasticity          : mean of the diagonal R[j, j] (how well each task is
                          learned right after training on it; lower is better).
                          Always report this next to forgetting: a frozen model
                          has zero forgetting but poor plasticity.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    avg_final = float(np.mean(R[T - 1, :]))
    forgetting, bwt = [], []
    for j in range(T - 1):
        forgetting.append(R[T - 1, j] - np.min(R[j: T - 1, j]))
        bwt.append(R[j, j] - R[T - 1, j])
    return {
        "average_performance": avg_final,
        "forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "backward_transfer": float(np.mean(bwt)) if bwt else 0.0,
        "plasticity": float(np.mean(np.diag(R))),
    }
