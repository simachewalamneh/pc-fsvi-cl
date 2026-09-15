import numpy as np


def standardized_error(y, mean_f, var_f, noise_std):
    denom = np.sqrt(var_f + noise_std ** 2)
    return (np.asarray(y) - mean_f) / denom


def task_change_correlation(z_scores, task_labels):
    z_scores = np.abs(np.asarray(z_scores))
    task_labels = np.asarray(task_labels)
    out = {}
    for t in np.unique(task_labels):
        out[f"task_{int(t)}_mean_abs_z"] = float(np.mean(z_scores[task_labels == t]))
    return out
