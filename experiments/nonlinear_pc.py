"""Stage 6: feature learning where predictive coding and backprop genuinely differ.
Target is non-linear; features are learned by BP (L-BFGS / Adam) or PC (Adam on
PC pseudo-gradients); a Bayesian last layer is then fit in closed form on top."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.nonlinear import (init_params, bp_loss_grad, pc_grad, cosine, train_lbfgs,
                           train_adam, bp_grad_only, BLL)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
H, SEEDS, EPOCHS, SIGMA = 30, [0, 1, 2, 3, 4], 1500, 0.1
target = lambda x: np.sin(3 * x) + 0.5 * x


def data(seed):
    rng = np.random.default_rng(100 + seed)
    X = np.sort(rng.uniform(-2, 2, 60)).reshape(-1, 1)
    y = target(X.ravel()) + rng.normal(0, SIGMA, 60)
    Xt = np.linspace(-2, 2, 200).reshape(-1, 1)
    return X, y, Xt, target(Xt.ravel())


def bll_rmse(th, X, y, Xt, ft):
    b = BLL(th, H, SIGMA); b.update(X, y)
    return float(np.sqrt(np.mean((ft - b.predict(Xt)[0]) ** 2)))


def pcg(K):
    return lambda th, X, y, s2, wd, H_: pc_grad(th, X, y, s2, wd, H_, K=K)


def main():
    rows = {}
    def add(name, r, info):
        rows.setdefault(name, []).append((r, info["time"], info["iters"], info["loss"]))

    for s in SEEDS:
        X, y, Xt, ft = data(s)
        th0 = init_params(H, seed=s)
        add("random features (no feature learning)", bll_rmse(th0, X, y, Xt, ft),
            dict(time=0.0, iters=0, loss=bp_loss_grad(th0, X, y, 1.0, 1e-2, H)[0]))
        th, info = train_lbfgs(th0, X, y, H, maxiter=500)
        add("BP + L-BFGS", bll_rmse(th, X, y, Xt, ft), info)
        th, info = train_adam(th0, X, y, H, bp_grad_only, epochs=EPOCHS)
        add("BP + Adam", bll_rmse(th, X, y, Xt, ft), info)
        th, info = train_adam(th0, X, y, H, pcg(20), epochs=EPOCHS)
        add("PC (K=20) + Adam", bll_rmse(th, X, y, Xt, ft), info)

    out = [f"=== Part A: test RMSE of BLL on learned features ({len(SEEDS)} seeds, n=60, H={H}) ===",
           f"{'Feature learner':42s} | {'RMSE':>13s} | {'train loss':>10s} | {'time (s)':>9s} | {'iters':>6s}"]
    for name, v in rows.items():
        a = np.array(v)
        out.append(f"{name:42s} | {a[:,0].mean():6.3f}+-{a[:,0].std():5.3f} | {a[:,3].mean():10.2f} | {a[:,1].mean():9.3f} | {int(a[:,2].mean()):6d}")

    # --- where PC differs: number of inference steps K
    out += ["", "=== Part B: PC vs BP as a function of inference steps K (Adam, same budget) ===",
            f"{'K':>4s} | {'cos(PC,BP) @init':>16s} | {'cos(PC,BP) @trained':>19s} | {'RMSE':>13s} | {'train loss':>10s}"]
    for K in (0, 1, 2, 5, 20, 100):
        cos0, cosT, rm, ls = [], [], [], []
        for s in SEEDS:
            X, y, Xt, ft = data(s)
            th0 = init_params(H, seed=s)
            thT, _ = train_lbfgs(th0, X, y, H, maxiter=500)
            for th, store in ((th0, cos0), (thT, cosT)):
                store.append(cosine(pc_grad(th, X, y, 1.0, 1e-2, H, K=K),
                                    bp_loss_grad(th, X, y, 1.0, 1e-2, H)[1]))
            th, info = train_adam(th0, X, y, H, pcg(K), epochs=EPOCHS)
            rm.append(bll_rmse(th, X, y, Xt, ft)); ls.append(info["loss"])
        out.append(f"{K:4d} | {np.mean(cos0):16.3f} | {np.mean(cosT):19.3f} | {np.mean(rm):6.3f}+-{np.std(rm):5.3f} | {np.mean(ls):10.2f}")
    out += ["", "Notes:",
            "- Last-layer fit is closed form for every row; differences come only from feature learning.",
            "- K=0: hidden layer gets no learning signal (e1=0), only weight decay -> features collapse.",
            "- cos @trained is measured at the BP optimum where the BP gradient ~ 0, so a low value means",
            "  PC's fixed point differs from the BP optimum (PC minimises the equilibrium energy, not the loss)."]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(RESULTS_DIR, "stage6_nonlinear.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
