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
    R = np.asarray(R)
    T = R.shape[0]
    avg_final = float(np.mean(R[T - 1, :]))
    # forgetting_j = R[T-1, j] - min_{i in [j, T-1]} R[i, j]  (error increase)
    forgetting = []
    for j in range(T - 1):
        forgetting.append(R[T - 1, j] - np.min(R[j : T - 1, j]))
    avg_forgetting = float(np.mean(forgetting)) if forgetting else 0.0
    # backward transfer (error increase on old tasks caused by new learning)
    bwt = []
    for j in range(T - 1):
        bwt.append(R[T - 1, j] - R[j, j])
    avg_bwt = float(np.mean(bwt)) if bwt else 0.0
    return {
        "average_performance": avg_final,
        "forgetting": avg_forgetting,
        "backward_transfer": avg_bwt,
    }
