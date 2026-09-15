import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import make_task, true_function, set_seed
from src.continual_learning import (
    sequential_gp_baseline,
    fsvi_continual,
    pc_fsvi_continual,
    ewc_like_baseline,
)
from src import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TASK_IDS = [1, 2, 3]


def run_all(seed=0, n_per_task=30, n_inducing=15):
    rng = set_seed(seed)
    task_data = [make_task(t, n=n_per_task, rng=rng) for t in TASK_IDS]
    Z = np.linspace(-3, 3, n_inducing).reshape(-1, 1)
    kw = dict(lengthscale=1.0, kernel_variance=1.0, noise_std=0.1)

    results = {}
    R_gp = sequential_gp_baseline(task_data)
    results["Sequential exact GP (oracle)"] = R_gp

    R_fsvi, model_fsvi = fsvi_continual(task_data, Z, kw, beta=1.0)
    results["FSVI continual (L-BFGS)"] = R_fsvi

    R_pc, model_pc, hists = pc_fsvi_continual(task_data, Z, kw, beta=1.0, n_iters=150, lr=1.0)
    results["PC-FSVI continual (proposed)"] = R_pc

    R_ewc, model_ewc = ewc_like_baseline(task_data, Z, kw, lam=10.0)
    results["EWC-like (point estimate)"] = R_ewc

    print(f"{'Method':32s} | {'AvgPerf':>8s} | {'Forget':>8s} | {'BWT':>8s}")
    for name, R in results.items():
        m = M.accuracy_matrix_to_metrics(R)
        print(f"{name:32s} | {m['average_performance']:8.4f} | {m['forgetting']:8.4f} | {m['backward_transfer']:8.4f}")
        print(f"  accuracy matrix (RMSE, rows=after task i, cols=eval task j):\n{np.round(R, 4)}")

    # --- Figure: predictive mean per task, before vs after full sequence ---
    xs = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for idx, t in enumerate(TASK_IDS):
        ax = axes[idx]
        f_true = true_function(t, xs)
        ax.plot(xs, f_true, "k--", label="true f" if idx == 0 else None)
        mean_gp, _ = None, None
        mean_pc, var_pc = model_pc.predict(xs)
        ax.plot(xs, mean_pc, label="PC-FSVI (final)" if idx == 0 else None, color="C0")
        std_pc = np.sqrt(var_pc)
        ax.fill_between(xs.ravel(), mean_pc - 2 * std_pc, mean_pc + 2 * std_pc, alpha=0.2, color="C0")
        Xi, yi, _ = task_data[idx]
        ax.scatter(Xi, yi, s=10, color="gray", alpha=0.6)
        ax.set_title(f"Task {t}")
    axes[0].legend()
    fig.suptitle("Stage 4: functional posterior after all tasks (PC-FSVI)")
    out_path = os.path.join(RESULTS_DIR, "stage4_final_posterior_per_task.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"figure saved to {out_path}")

    # --- Figure: forgetting comparison across methods ---
    plt.figure(figsize=(6.5, 4))
    names = list(results.keys())
    forget_vals = [M.accuracy_matrix_to_metrics(results[n])["forgetting"] for n in names]
    plt.barh(names, forget_vals, color=["C2", "C0", "C1", "C3"])
    plt.xlabel("Forgetting (RMSE increase on old tasks)")
    plt.title("Stage 4: forgetting across methods")
    out_path2 = os.path.join(RESULTS_DIR, "stage4_forgetting_comparison.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"figure saved to {out_path2}")

    return results, task_data, (model_fsvi, model_pc, model_ewc)


if __name__ == "__main__":
    run_all()
