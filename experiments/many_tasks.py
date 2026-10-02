"""Stage 7: many tasks, recurring functions -> cost of component growth.
Task sequence of length T drawn from a pool of 4 distinct functions.
Methods on a frozen-feature Bayesian last layer:
  merge    : one component, plain chaining (exact Bayes, interferes)
  spawn    : one component per task (no interference, no sharing, memory ~ T)
  evidence : merge iff evidence beats a fresh prior (memory ~ #distinct functions)
Feature variants: random / BP-pretrained / PC-pretrained (on an unrelated aux function)."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.nonlinear import (init_params, train_adam, bp_grad_only, pc_grad, bll_continual)
from src import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

H, NOISE, N_PER_TASK = 30, 0.2, 10
SEEDS = [0, 1, 2, 3, 4]
T_LIST = [6, 12, 24, 48]
POOL = [np.sin, np.cos, lambda x: -np.sin(x), lambda x: 0.4 * x]
GRID = np.linspace(-3, 3, 100).reshape(-1, 1)


def make_sequence(T, seed):
    rng = np.random.default_rng(seed)
    ids = list(range(len(POOL))) + list(rng.integers(0, len(POOL), T - len(POOL)))
    ids = [int(i) for i in rng.permutation(ids)]
    td, ev = [], []
    for k in ids:
        x = np.sort(rng.uniform(-3, 3, N_PER_TASK))
        f = POOL[k](x)
        td.append((x.reshape(-1, 1), f + rng.normal(0, NOISE, N_PER_TASK), f))
        ev.append((GRID, POOL[k](GRID).ravel()))
    return td, ev, ids


def pretrained_features(variant, seed):
    th0 = init_params(H, seed=seed)
    if variant == "random":
        return th0
    rng = np.random.default_rng(900 + seed)
    X = np.sort(rng.uniform(-3, 3, 80)).reshape(-1, 1)
    y = np.sin(2 * X.ravel()) + 0.5 * np.cos(4 * X.ravel()) + rng.normal(0, 0.1, 80)
    gfn = bp_grad_only if variant == "BP-pretrained" else (lambda th, X, y, s2, wd, H_: pc_grad(th, X, y, s2, wd, H_, K=20))
    return train_adam(th0, X, y, H, gfn, epochs=1500)[0]


def purity(assign, ids):
    comps = {}
    for a, k in zip(assign, ids):
        comps.setdefault(a, []).append(k)
    return sum(max(np.bincount(v)) for v in comps.values()) / len(ids)


def main():
    lines = [f"=== Stage 7: many tasks (pool of {len(POOL)} functions, n={N_PER_TASK}/task, noise={NOISE}, {len(SEEDS)} seeds) ===",
             "Memory = floats stored for component posteriors, (H+1)+(H+1)^2 per component.",
             f"{'features':14s} {'T':>3s} {'method':9s} | {'AvgPerf':>13s} | {'BWT':>7s} | {'#comps':>6s} | {'memory':>7s} | {'purity':>6s} | {'time (s)':>8s}"]
    D = H + 1
    for variant in ("random", "BP-pretrained", "PC-pretrained"):
        ths = {s: pretrained_features(variant, s) for s in SEEDS}
        for T in T_LIST:
            for mode in ("merge", "spawn", "evidence"):
                ap, bwt, nc, pu, tt = [], [], [], [], []
                for s in SEEDS:
                    td, ev, ids = make_sequence(T, s)
                    R, comps, assign, sec = bll_continual(td, ths[s], H, mode=mode, noise_std=NOISE, eval_data=ev)
                    m = M.accuracy_matrix_to_metrics(R)
                    ap.append(m["average_performance"]); bwt.append(m["backward_transfer"])
                    nc.append(len(comps)); pu.append(purity(assign, ids)); tt.append(sec)
                lines.append(f"{variant:14s} {T:3d} {mode:9s} | {np.mean(ap):6.3f}+-{np.std(ap):5.3f} | {np.mean(bwt):7.3f} | "
                             f"{np.mean(nc):6.1f} | {int(np.mean(nc) * (D + D * D)):7d} | {np.mean(pu):6.2f} | {np.mean(tt):8.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS_DIR, "stage7_many_tasks.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
