import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import make_task
from src.gp import ExactGP
from src.variational import FSVI


def test_fsvi_matches_exact_gp():
 
    rng = np.random.default_rng(1)
    X, y, f = make_task(1, n=25, rng=rng)

    gp = ExactGP(noise_std=0.1).fit(X, y)
    mean_gp, var_gp = gp.predict(X)

    Z = np.linspace(-3, 3, 15).reshape(-1, 1)
    fsvi = FSVI(Z, noise_std=0.1)
    fsvi.fit(X, y, beta=1.0, maxiter=3000)
    mean_v, var_v = fsvi.predict(X)

    assert np.max(np.abs(mean_gp - mean_v)) < 1e-2
    assert np.max(np.abs(var_gp - var_v)) < 5e-2


def test_kl_zero_when_q_equals_prior():
    Z = np.linspace(-3, 3, 8).reshape(-1, 1)
    fsvi = FSVI(Z, noise_std=0.1)
    # free energy at the prior should be finite and the KL term (isolated)
    # should vanish when m=0, L = chol(Kzz)  (fsvi is initialised to the prior)
    m, L = fsvi.m, fsvi.L
    S = L @ L.T
    prior_cov_inv = np.linalg.inv(fsvi.Kzz + 1e-8 * np.eye(fsvi.M))
    sign0, logdet0 = np.linalg.slogdet(fsvi.Kzz)
    sign1, logdet1 = np.linalg.slogdet(S + 1e-10 * np.eye(fsvi.M))
    kl = 0.5 * (np.trace(prior_cov_inv @ S) + 0 - fsvi.M + logdet0 - logdet1)
    assert abs(kl) < 1e-4


if __name__ == "__main__":
    test_fsvi_matches_exact_gp()
    test_kl_zero_when_q_equals_prior()
    print("test_gp_fsvi.py: all tests passed")
