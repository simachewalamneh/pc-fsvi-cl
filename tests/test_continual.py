import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import make_task
from src.continual_learning import pc_fsvi_continual, sequential_gp_baseline
from src import metrics as M


def test_pc_fsvi_matches_sequential_gp_oracle():
    rng = np.random.default_rng(0)
    task_data = [make_task(t, n=25, rng=rng) for t in [1, 2, 3]]
    Z = np.linspace(-3, 3, 20).reshape(-1, 1)
    kw = dict(lengthscale=1.0, kernel_variance=1.0, noise_std=0.1)

    R_pc, _, _ = pc_fsvi_continual(task_data, Z, kw, beta=1.0, n_iters=50, lr=1.0)
    R_gp = sequential_gp_baseline(task_data)

    diff = np.nanmax(np.abs(R_pc - R_gp))
    assert diff < 0.05


def test_forgetting_metric_zero_for_oracle_on_disjoint_inputs():
    R = np.array([
        [0.1, np.nan, np.nan],
        [0.1, 0.2, np.nan],
        [0.1, 0.2, 0.3],
    ])
    out = M.accuracy_matrix_to_metrics(R)
    assert out["forgetting"] == 0.0
    assert out["backward_transfer"] == 0.0
    assert abs(out["average_performance"] - 0.2) < 1e-9


if __name__ == "__main__":
    test_pc_fsvi_matches_sequential_gp_oracle()
    test_forgetting_metric_zero_for_oracle_on_disjoint_inputs()
    print("test_continual.py: all tests passed")
