import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import make_task
from src.variational import FSVI
from src.predictive_coding import pc_infer, closed_form_optimum


def test_pc_converges_to_closed_form():
    rng = np.random.default_rng(2)
    X, y, f = make_task(1, n=30, rng=rng)
    Z = np.linspace(-3, 3, 12).reshape(-1, 1)
    fsvi = FSVI(Z, noise_std=0.1)
    prior_mean = np.zeros(fsvi.M)
    prior_cov = fsvi.Kzz.copy()

    m_pc, S_pc, hist = pc_infer(
        fsvi, X, y, prior_mean, prior_cov, beta=1.0, n_iters=50, lr=1.0
    )
    m_cf, S_cf = closed_form_optimum(fsvi, X, y, prior_mean, prior_cov, beta=1.0)

    assert np.max(np.abs(m_pc - m_cf)) < 1e-6
    assert np.max(np.abs(S_pc - S_cf)) < 1e-6
    # error norm should be (weakly) decreasing over iterations
    assert hist["error_norm"][-1] <= hist["error_norm"][0] + 1e-8


def test_beta_controls_stability():
    """Larger beta (stronger KL pull to the prior/previous posterior)
    should keep the posterior mean closer to the prior mean."""
    rng = np.random.default_rng(3)
    X, y, f = make_task(1, n=30, rng=rng)
    Z = np.linspace(-3, 3, 12).reshape(-1, 1)
    fsvi = FSVI(Z, noise_std=0.1)
    prior_mean = np.zeros(fsvi.M)
    prior_cov = fsvi.Kzz.copy()

    m_low, _ = closed_form_optimum(fsvi, X, y, prior_mean, prior_cov, beta=0.1)
    m_high, _ = closed_form_optimum(fsvi, X, y, prior_mean, prior_cov, beta=10.0)

    dist_low = np.linalg.norm(m_low - prior_mean)
    dist_high = np.linalg.norm(m_high - prior_mean)
    assert dist_high < dist_low


if __name__ == "__main__":
    test_pc_converges_to_closed_form()
    test_beta_controls_stability()
    print("test_predictive_coding.py: all tests passed")
