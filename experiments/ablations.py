"""Stage 5: ablations, multi-seed, held-out grid, two benchmarks.
Reports mean +- std over seeds for AvgPerf / Forget / BWT / Plasticity."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import make_benchmark
from src.continual_learning import (
    pc_fsvi_continual, sequential_gp_baseline, ewc_like_baseline, surprise_beta,
    spawn_continual, independent_baseline,
)
from src import metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
BASE_KW = dict(lengthscale=1.0, kernel_variance=1.0, noise_std=0.1)
Z = np.linspace(-3, 3, 15).reshape(-1, 1)


def fixed(b):
    return lambda i, model, X, y: b


def frozen_after_first(i, model, X, y):
    return 1.0 if i == 0 else 1e6


def configs():
    """(label, kernel_kwargs, runner(task_data, eval_data, kw) -> R)"""
    def pc(beta=1.0, **extra):
        return lambda td, ev, kw: pc_fsvi_continual(td, Z, kw, beta=beta, eval_data=ev, **extra)[0]

    cfgs = [
        ("reference: sequential exact GP", BASE_KW,
         lambda td, ev, kw: sequential_gp_baseline(td, eval_data=ev, noise_std=kw["noise_std"],
                                                   lengthscale=kw["lengthscale"])),
        ("reference: EWC-like (accumulated Fisher)", BASE_KW,
         lambda td, ev, kw: ewc_like_baseline(td, Z, kw, lam=10.0, accumulate=True, eval_data=ev)[0]),
        ("reference: frozen after task 1", BASE_KW, pc(frozen_after_first)),
        ("precision: fixed 1/sigma_n^2", BASE_KW, pc(fixed(1.0))),
        ("precision: adaptive 1/(sigma_n^2+var_f)", BASE_KW, pc(fixed(1.0), adaptive_precision=True)),
        ("precision: Pi=1 (rescales likelihood; NOT a pure ablation)", BASE_KW,
         pc(fixed(1.0), precision_override=1.0)),
        ("reference: independent models (no memory)", BASE_KW,
         lambda td, ev, kw: independent_baseline(td, Z, kw, eval_data=ev)),
        ("PROPOSED: detect-and-spawn (thr=3)", BASE_KW,
         lambda td, ev, kw: spawn_continual(td, Z, kw, thr=3.0, eval_data=ev)[0]),
        ("detect-and-spawn (thr=1.5)", BASE_KW,
         lambda td, ev, kw: spawn_continual(td, Z, kw, thr=1.5, eval_data=ev)[0]),
        ("detect-and-spawn (thr=10)", BASE_KW,
         lambda td, ev, kw: spawn_continual(td, Z, kw, thr=10.0, eval_data=ev)[0]),
        ("beta=1e-3 (almost no stability)", BASE_KW, pc(fixed(1e-3))),
        ("beta=0.3", BASE_KW, pc(fixed(0.3))),
        ("beta=1", BASE_KW, pc(fixed(1.0))),
        ("beta=3", BASE_KW, pc(fixed(3.0))),
        ("beta=10", BASE_KW, pc(fixed(10.0))),
        ("adaptive beta: surprise-based", BASE_KW, pc(surprise_beta)),
    ]
    for ls in (0.5, 1.0, 2.0):
        cfgs.append((f"lengthscale={ls} (beta=1)", {**BASE_KW, "lengthscale": ls}, pc(fixed(1.0))))
    for ns in (0.05, 0.1, 0.3):
        cfgs.append((f"noise_std={ns} (beta=1)", {**BASE_KW, "noise_std": ns}, pc(fixed(1.0))))
    return cfgs


def spawn_cfgs():
    sp = lambda **k: (lambda td, ev, kw: spawn_continual(td, Z, kw, eval_data=ev, **k)[0])
    return [
        ("always-merge (thr=inf = plain chaining)", BASE_KW, sp(thr=np.inf)),
        ("always-spawn (thr=0)", BASE_KW, sp(thr=0.0)),
        ("spawn: surprise thr=1.5", BASE_KW, sp(thr=1.5)),
        ("spawn: surprise thr=3", BASE_KW, sp(thr=3.0)),
        ("spawn: surprise thr=10", BASE_KW, sp(thr=10.0)),
        ("spawn: evidence (Bayes factor, no threshold)", BASE_KW, sp(criterion="evidence")),
    ]


def focus_configs():
    base = {c[0]: c for c in configs()}
    keep = ["reference: sequential exact GP", "reference: EWC-like (accumulated Fisher)",
            "reference: frozen after task 1", "beta=3"]
    return [base[k] for k in keep] + spawn_cfgs()


N_PER_TASK = {"conflicting": 30, "regional": 30, "offset": 30, "halfconflict": 15, "revisit": 10}


def run_benchmark(name):
    cfgs = configs() if name in ("conflicting", "regional") else focus_configs()
    table = {label: [] for label, _, _ in cfgs}
    for seed in SEEDS:
        td, ev = make_benchmark(name, seed=seed, n=N_PER_TASK[name])
        for label, kw, runner in cfgs:
            table[label].append(M.accuracy_matrix_to_metrics(runner(td, ev, kw)))
    lines = [f"=== benchmark: {name} (mean +- std over {len(SEEDS)} seeds, held-out grid) ===",
             f"{'Ablation':60s} | {'AvgPerf':>14s} | {'Forget':>14s} | {'BWT':>14s} | {'Plasticity':>14s}"]
    for label, _, _ in cfgs:
        cells = []
        for key in ("average_performance", "forgetting", "backward_transfer", "plasticity"):
            v = np.array([r[key] for r in table[label]])
            cells.append(f"{v.mean():6.3f}+-{v.std():5.3f}")
        lines.append(f"{label:60s} | " + " | ".join(f"{c:>14s}" for c in cells))
    return "\n".join(lines)


def main():
    out = []
    for name in ("conflicting", "regional", "offset", "halfconflict", "revisit"):
        block = run_benchmark(name)
        print(block, "\n")
        out.append(block)
    note = ("Notes: BWT is signed (negative = old tasks got worse). Plasticity = mean diagonal RMSE.\n"
            "Forgetting alone is misleading: 'frozen after task 1' has zero forgetting.")
    print(note)
    path = os.path.join(RESULTS_DIR, "stage5_ablations.txt")
    with open(path, "w") as fh:
        fh.write("\n\n".join(out) + "\n\n" + note + "\n")
    print(f"results saved to {path}")


if __name__ == "__main__":
    main()
