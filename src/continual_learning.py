import numpy as np
from .variational import FSVI
from .predictive_coding import pc_infer
from .gp import ExactGP
from . import metrics as M


def _eval_on_tasks(predict_fn, tasks_xy):
    """tasks_xy: list of (X, y, f_true) per task. Returns list of RMSE."""
    out = []
    for X, y, f_true in tasks_xy:
        mean, var = predict_fn(X)
        out.append(M.rmse(f_true, mean))
    return out


def independent_baseline(task_data, Z, kernel_kwargs, beta=1.0, maxiter=3000):
    T = len(task_data)
    R = np.zeros((T, T))
    for i, (Xi, yi, _) in enumerate(task_data):
        model = FSVI(Z, **kernel_kwargs)
        model.fit(Xi, yi, beta=beta, prior_mean=None, prior_cov=None, maxiter=maxiter)
        for j in range(i + 1):
            Xj, _, fj = task_data[j]
            mean, _ = model.predict(Xj)
            R[i, j] = M.rmse(fj, mean) if j == i else np.nan  # independent: no cross-task model
    # independent models can't be evaluated on other tasks -> only diagonal meaningful
    return R


def sequential_gp_baseline(task_data):
    T = len(task_data)
    R = np.full((T, T), np.nan)
    X_all, y_all = None, None
    for i, (Xi, yi, _) in enumerate(task_data):
        X_all = Xi if X_all is None else np.vstack([X_all, Xi])
        y_all = yi if y_all is None else np.concatenate([y_all, yi])
        gp = ExactGP(noise_std=0.1)
        gp.fit(X_all, y_all)
        for j in range(i + 1):
            Xj, _, fj = task_data[j]
            mean, _ = gp.predict(Xj)
            R[i, j] = M.rmse(fj, mean)
    return R


def fsvi_continual(task_data, Z, kernel_kwargs, beta=1.0, maxiter=3000):
    T = len(task_data)
    R = np.full((T, T), np.nan)
    model = FSVI(Z, **kernel_kwargs)
    prior_mean, prior_cov = None, None
    for i, (Xi, yi, _) in enumerate(task_data):
        model.fit(Xi, yi, beta=beta, prior_mean=prior_mean, prior_cov=prior_cov, maxiter=maxiter)
        prior_mean, prior_cov = model.posterior()
        for j in range(i + 1):
            Xj, _, fj = task_data[j]
            mean, _ = model.predict(Xj)
            R[i, j] = M.rmse(fj, mean)
    return R, model


def pc_fsvi_continual(
    task_data, Z, kernel_kwargs, beta=1.0, n_iters=200, lr=0.05, adaptive_precision=False
):
    T = len(task_data)
    R = np.full((T, T), np.nan)
    model = FSVI(Z, **kernel_kwargs)
    prior_mean = np.zeros(model.M)
    prior_cov = model.Kzz.copy()
    histories = []
    for i, (Xi, yi, _) in enumerate(task_data):
        m, S, hist = pc_infer(
            model, Xi, yi, prior_mean, prior_cov, beta=beta,
            n_iters=n_iters, lr=lr, adaptive_precision=adaptive_precision,
        )
        model.m, model.L = m, np.linalg.cholesky(S + 1e-8 * np.eye(model.M))
        histories.append(hist)
        prior_mean, prior_cov = model.posterior()
        for j in range(i + 1):
            Xj, _, fj = task_data[j]
            mean, _ = model.predict(Xj)
            R[i, j] = M.rmse(fj, mean)
    return R, model, histories


def ewc_like_baseline(task_data, Z, kernel_kwargs, lam=10.0, maxiter=3000):
 
    from scipy.optimize import minimize as spminimize

    T = len(task_data)
    R = np.full((T, T), np.nan)
    model = FSVI(Z, **kernel_kwargs)
    m_star = np.zeros(model.M)
    Fisher = np.zeros(model.M)  # diagonal Fisher, 0 for task 1 (no penalty)
    sigma_n2 = model.noise_std ** 2

    def A_matrix(X):
        Kxz = model._Kxz(X)
        return Kxz @ model.Kzz_inv

    for i, (Xi, yi, _) in enumerate(task_data):
        A = A_matrix(Xi)

        def neg_log_post(m):
            resid = yi - A @ m
            nll = 0.5 * np.sum(resid ** 2) / sigma_n2
            ewc_pen = 0.5 * lam * np.sum(Fisher * (m - m_star) ** 2)
            return nll + ewc_pen

        res = spminimize(neg_log_post, m_star.copy(), method="L-BFGS-B",
                          options={"maxiter": maxiter})
        m_star = res.x
        model.m = m_star
        model.L = np.eye(model.M) * 1e-3  # point estimate: negligible covariance
        # diagonal Fisher = diag(A^T A) / sigma_n^2 (Gauss-Newton approx)
        Fisher = np.sum(A ** 2, axis=0) / sigma_n2
        for j in range(i + 1):
            Xj, _, fj = task_data[j]
            mean, _ = model.predict(Xj)
            R[i, j] = M.rmse(fj, mean)
    return R, model
