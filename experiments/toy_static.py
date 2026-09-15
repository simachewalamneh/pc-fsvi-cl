import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import make_task, true_function, set_seed
from src.gp import ExactGP

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main(seed=0):
    rng = set_seed(seed)
    X, y, f = make_task(1, n=25, noise_std=0.1, rng=rng)

    gp = ExactGP(lengthscale=1.0, variance=1.0, noise_std=0.1)
    gp.fit(X, y)

    xs = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    mean, var = gp.predict(xs)
    std = np.sqrt(var)
    f_true = true_function(1, xs)

    rmse = float(np.sqrt(np.mean((mean - f_true) ** 2)))
    print(f"[Stage 1] Static GP posterior mean RMSE vs. true function: {rmse:.4f}")
    print(f"[Stage 1] neg-log-marginal-likelihood: {gp.neg_log_marginal_likelihood():.4f}")

    plt.figure(figsize=(7, 4.5))
    prior_samples = gp.sample_prior(xs, n_samples=3, rng=rng)
    for s in prior_samples:
        plt.plot(xs, s, color="gray", alpha=0.3, lw=1)
    plt.fill_between(xs.ravel(), mean - 2 * std, mean + 2 * std, alpha=0.25, label="95% CI")
    plt.plot(xs, mean, label="GP posterior mean", color="C0")
    plt.plot(xs, f_true, "--", label="true function", color="C1")
    plt.scatter(X, y, color="k", s=15, zorder=5, label="observations")
    plt.legend()
    plt.title("Stage 1: Exact GP regression, y = sin(x) + noise")
    out_path = os.path.join(RESULTS_DIR, "stage1_static_gp.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Stage 1] figure saved to {out_path}")


if __name__ == "__main__":
    main()
