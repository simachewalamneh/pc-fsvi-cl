import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import make_task, set_seed
from src.variational import FSVI
from src.predictive_coding import pc_infer
from src import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TASK_IDS = [1, 2, 3]


def run_continual(task_data, Z, kw, beta_schedule, adaptive_precision=False,
                   precision_override=None, n_iters=150, lr=1.0):
    T = len(task_data)
    R = np.full((T, T), np.nan)
    model = FSVI(Z, **kw)
    prior_mean = np.zeros(model.M)
    prior_cov = model.Kzz.copy()
    for i, (Xi, yi, _) in enumerate(task_data):
        beta = beta_schedule(i, model, Xi)
        m, S, _ = pc_infer(model, Xi, yi, prior_mean, prior_cov, beta=beta,
                            n_iters=n_iters, lr=lr, adaptive_precision=adaptive_precision,
                            precision_override=precision_override)
        model.m, model.L = m, np.linalg.cholesky(S + 1e-8 * np.eye(model.M))
        prior_mean, prior_cov = model.posterior()
        for j in range(i + 1):
            Xj, _, fj = task_data[j]
            mean, _ = model.predict(Xj)
            R[i, j] = M.rmse(fj, mean)
    return R


def main(seed=0):
    rng = set_seed(seed)
    task_data = [make_task(t, n=30, rng=rng) for t in TASK_IDS]
    Z = np.linspace(-3, 3, 15).reshape(-1, 1)
    base_kw = dict(lengthscale=1.0, kernel_variance=1.0, noise_std=0.1)

    rows = []

    # 1. precision weighting on/off (beta fixed = 1)
    R_pi = run_continual(task_data, Z, base_kw, lambda i, m, X: 1.0)
    rows.append(("precision: 1/sigma_n^2 (on)", M.accuracy_matrix_to_metrics(R_pi)))
    # "off" = precision forced to 1.0 for ALL points, independent of
    # sigma_n^2, via the explicit precision_override toggle (isolates the
    # weighting itself -- nothing else about the model changes).
    R_nopi = run_continual(task_data, Z, base_kw, lambda i, m, X: 1.0, precision_override=1.0)
    rows.append(("precision: unweighted (Pi=1)", M.accuracy_matrix_to_metrics(R_nopi)))

    # 2. KL stability term on/off
    R_beta1 = run_continual(task_data, Z, base_kw, lambda i, m, X: 1.0)
    rows.append(("beta=1 (stability on)", M.accuracy_matrix_to_metrics(R_beta1)))
    R_beta0 = run_continual(task_data, Z, base_kw, lambda i, m, X: 0.0)
    rows.append(("beta=0 (no stability)", M.accuracy_matrix_to_metrics(R_beta0)))

    # 3. fixed vs adaptive beta
    def adaptive_beta(i, model, X):
        if i == 0:
            return 1.0
        _, var = model.predict(X)
        mean_var = float(np.mean(var))
        # higher pre-task predictive uncertainty on the new inputs -> the
        # model already "doesn't know" this region -> lean more plastic
        return float(np.clip(1.0 / (mean_var + 1e-3), 0.1, 10.0))

    R_adapt = run_continual(task_data, Z, base_kw, adaptive_beta)
    rows.append(("adaptive beta_t (uncertainty-based)", M.accuracy_matrix_to_metrics(R_adapt)))

    # 4. kernel lengthscale sweep
    for ls in [0.5, 1.0, 2.0]:
        kw = {**base_kw, "lengthscale": ls}
        R = run_continual(task_data, Z, kw, lambda i, m, X: 1.0)
        rows.append((f"lengthscale={ls}", M.accuracy_matrix_to_metrics(R)))

    # 5. noise level sweep
    for ns in [0.05, 0.1, 0.3]:
        kw = {**base_kw, "noise_std": ns}
        R = run_continual(task_data, Z, kw, lambda i, m, X: 1.0)
        rows.append((f"noise_std={ns}", M.accuracy_matrix_to_metrics(R)))

    print(f"{'Ablation':38s} | {'AvgPerf':>8s} | {'Forget':>8s} | {'BWT':>8s}")
    for name, m in rows:
        print(f"{name:38s} | {m['average_performance']:8.4f} | {m['forgetting']:8.4f} | {m['backward_transfer']:8.4f}")

    out_path = os.path.join(RESULTS_DIR, "stage5_ablations.txt")
    with open(out_path, "w") as fh:
        fh.write(f"{'Ablation':38s} | {'AvgPerf':>8s} | {'Forget':>8s} | {'BWT':>8s}\n")
        for name, m in rows:
            fh.write(f"{name:38s} | {m['average_performance']:8.4f} | {m['forgetting']:8.4f} | {m['backward_transfer']:8.4f}\n")
    print(f"results saved to {out_path}")


if __name__ == "__main__":
    main()
