import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import make_task, true_function, set_seed
from src.gp import ExactGP
from src.variational import FSVI
from src.predictive_coding import pc_infer
from src import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main(seed=0):
    rng = set_seed(seed)
    X, y, f = make_task(1, n=30, noise_std=0.1, rng=rng)
    xs = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    f_true = true_function(1, xs)
    Z = np.linspace(-3, 3, 15).reshape(-1, 1)
    kw = dict(lengthscale=1.0, kernel_variance=1.0, noise_std=0.1)

    # --- A: exact GP ---
    gp = ExactGP(noise_std=0.1).fit(X, y)
    mean_A, var_A = gp.predict(xs)

    # --- B: FSVI / batch L-BFGS ---
    t0 = time.time()
    fsvi_B = FSVI(Z, **kw)
    fsvi_B.fit(X, y, beta=1.0, maxiter=3000)
    t_B = time.time() - t0
    mean_B, var_B = fsvi_B.predict(xs)

    # --- C: FSVI / predictive coding ---
    t0 = time.time()
    fsvi_C = FSVI(Z, **kw)
    prior_mean = np.zeros(fsvi_C.M)
    prior_cov = fsvi_C.Kzz.copy()
    m_C, S_C, hist_C = pc_infer(fsvi_C, X, y, prior_mean, prior_cov, beta=1.0, n_iters=100, lr=1.0)
    fsvi_C.m, fsvi_C.L = m_C, np.linalg.cholesky(S_C + 1e-10 * np.eye(fsvi_C.M))
    t_C = time.time() - t0
    mean_C, var_C = fsvi_C.predict(xs)

    print("Method        | RMSE vs true f | wall time (s)")
    for name, mean, t in [("A: exact GP", mean_A, None), ("B: FSVI/L-BFGS", mean_B, t_B), ("C: FSVI/PC", mean_C, t_C)]:
        rmse = M.rmse(f_true, mean)
        t_str = f"{t:.4f}" if t is not None else "  -   "
        print(f"{name:14s} | {rmse:.4f}          | {t_str}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, f_true, "--", color="k", label="true function")
    plt.plot(xs, mean_A, label="A: exact GP", lw=2)
    plt.plot(xs, mean_B, label="B: FSVI (L-BFGS)", lw=2, linestyle=":")
    plt.plot(xs, mean_C, label="C: FSVI (predictive coding)", lw=2, linestyle="-.")
    plt.scatter(X, y, color="k", s=12, alpha=0.6, label="observations")
    plt.legend()
    plt.title("Stage 2-3: exact GP vs. FSVI (L-BFGS) vs. FSVI (predictive coding)")
    out_path = os.path.join(RESULTS_DIR, "stage2_3_methods_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"figure saved to {out_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(hist_C["error_norm"])
    plt.xlabel("predictive-coding iteration")
    plt.ylabel("RMS prediction error")
    plt.title("Predictive-coding inference loop convergence")
    out_path2 = os.path.join(RESULTS_DIR, "stage3_pc_convergence.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"figure saved to {out_path2}")


if __name__ == "__main__":
    main()
