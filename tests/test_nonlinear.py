import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.nonlinear import (init_params, bp_loss_grad, pc_grad, cosine, BLL)


def test_bp_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    H = 6
    X = rng.uniform(-2, 2, (15, 1)); y = np.sin(X.ravel())
    th = init_params(H, seed=1)
    _, g = bp_loss_grad(th, X, y, 1.0, 1e-2, H)
    num = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th); e[i] = 1e-6
        num[i] = (bp_loss_grad(th + e, X, y, 1.0, 1e-2, H)[0] - bp_loss_grad(th - e, X, y, 1.0, 1e-2, H)[0]) / 2e-6
    assert np.max(np.abs(g - num)) < 1e-5


def test_pc_approaches_bp_with_more_inference_steps():
    rng = np.random.default_rng(1)
    H = 20
    X = rng.uniform(-2, 2, (40, 1)); y = np.sin(3 * X.ravel())
    th = init_params(H, seed=2)
    g_bp = bp_loss_grad(th, X, y, 1.0, 1e-2, H)[1]
    c0 = cosine(pc_grad(th, X, y, 1.0, 1e-2, H, K=0), g_bp)
    c50 = cosine(pc_grad(th, X, y, 1.0, 1e-2, H, K=50, eta=0.05), g_bp)
    assert c50 > c0 and c50 > 0.8


def test_bll_chaining_equals_batch():
    rng = np.random.default_rng(2)
    H = 10
    th = init_params(H, seed=3)
    X = rng.uniform(-3, 3, (30, 1)); y = np.sin(X.ravel()) + rng.normal(0, 0.1, 30)
    a = BLL(th, H); a.update(X[:12], y[:12]); a.update(X[12:], y[12:])
    b = BLL(th, H); b.update(X, y)
    assert np.max(np.abs(a.m - b.m)) < 1e-6 and np.max(np.abs(a.S - b.S)) < 1e-8


if __name__ == "__main__":
    test_bp_gradient_matches_finite_difference()
    test_pc_approaches_bp_with_more_inference_steps()
    test_bll_chaining_equals_batch()
    print("test_nonlinear.py: all tests passed")
