"""Non-linear model: tanh feature net + Bayesian last layer (BLL).

Two separate inference problems, deliberately kept apart:
  1. FEATURE LEARNING (non-linear, non-convex): backprop (L-BFGS / Adam) vs
     predictive coding (iterative inference of hidden activities + local
     weight updates). Here PC and BP genuinely differ.
  2. LAST-LAYER POSTERIOR (linear-Gaussian): closed form. PC = Newton step =
     exact Bayes here, exactly as in the FSVI toy.
"""
import time
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------------ params
def _unpack(th, H, d):
    i = H * d
    return th[:i].reshape(H, d), th[i:i + H], th[i + H:i + 2 * H], th[i + 2 * H]


def _pack(W1, b1, w2, b2):
    return np.concatenate([W1.ravel(), b1, w2, [b2]])


def init_params(H=30, d=1, seed=0, scale=1.5):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, scale, (H, d))
    b1 = rng.uniform(-2, 2, H)
    w2 = rng.normal(0, 1 / np.sqrt(H), H)
    return _pack(W1, b1, w2, 0.0)


def phi(th, X, H, d=1):
    """Features with a constant bias column: (N, H+1)."""
    W1, b1, _, _ = _unpack(th, H, d)
    return np.hstack([np.tanh(X @ W1.T + b1), np.ones((len(X), 1))])


# ------------------------------------------------- gradients: BP vs PC
def bp_loss_grad(th, X, y, s2, wd, H, d=1):
    """Loss = 0.5*sum((out-y)^2)/s2 + 0.5*wd*(|W1|^2+|w2|^2) and its gradient."""
    W1, b1, w2, b2 = _unpack(th, H, d)
    h = np.tanh(X @ W1.T + b1)
    res = h @ w2 + b2 - y
    loss = 0.5 * np.sum(res ** 2) / s2 + 0.5 * wd * (np.sum(W1 ** 2) + np.sum(w2 ** 2))
    r = res / s2
    gw2 = h.T @ r + wd * w2
    dz = (r[:, None] * w2[None, :]) * (1 - h ** 2)
    return loss, _pack(dz.T @ X + wd * W1, dz.sum(0), gw2, r.sum())


def pc_grad(th, X, y, s2, wd, H, d=1, K=20, eta=0.2):
    """Predictive-coding pseudo-gradient (same sign/scale as bp_loss_grad).
    Energy: 0.5|x1 - mu1|^2 + 0.5|y - out(x1)|^2/s2, x0=X and x2=y clamped.
      inference (K steps): dx1 = -e1 + f'(x1) * (w2 * e2)
      learning (local)   : dW1 ~ -e1 x0^T,  dw2 ~ -e2 f(x1)
    K=0 -> e1=0 -> hidden layer receives NO learning signal.
    K->inf (small error) -> approaches backprop."""
    W1, b1, w2, b2 = _unpack(th, H, d)
    mu1 = X @ W1.T + b1
    x1 = mu1.copy()
    for _ in range(K):
        h = np.tanh(x1)
        e2 = (y - (h @ w2 + b2)) / s2
        x1 = x1 + eta * (-(x1 - mu1) + (e2[:, None] * w2[None, :]) * (1 - h ** 2))
    h = np.tanh(x1)
    e2 = (y - (h @ w2 + b2)) / s2
    e1 = x1 - mu1
    return _pack(-(e1.T @ X) + wd * W1, -e1.sum(0), -(h.T @ e2) + wd * w2, -e2.sum())


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------ optimisers
def train_lbfgs(th0, X, y, H, s2=1.0, wd=1e-2, maxiter=500):
    t0 = time.perf_counter()
    res = minimize(bp_loss_grad, th0, args=(X, y, s2, wd, H), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x, dict(time=time.perf_counter() - t0, iters=int(res.nit),
                       loss=float(res.fun))


def train_adam(th0, X, y, H, gradfn, s2=1.0, wd=1e-2, epochs=1500, lr=0.01):
    th = th0.copy()
    m = np.zeros_like(th); v = np.zeros_like(th)
    b1, b2, eps, N = 0.9, 0.999, 1e-8, len(X)
    t0 = time.perf_counter()
    for t in range(1, epochs + 1):
        g = gradfn(th, X, y, s2, wd, H) / N
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        th -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    loss = bp_loss_grad(th, X, y, s2, wd, H)[0]
    return th, dict(time=time.perf_counter() - t0, iters=epochs, loss=float(loss))


def bp_grad_only(th, X, y, s2, wd, H):
    return bp_loss_grad(th, X, y, s2, wd, H)[1]


# -------------------------------------------------- Bayesian last layer
def _chol(C, jitter=1e-8):
    C = 0.5 * (C + C.T)
    for k in range(12):
        try:
            return np.linalg.cholesky(C + jitter * 10.0 ** k * np.eye(len(C)))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("not PD")


class BLL:
    """q(w)=N(m,S) over last-layer weights on FROZEN features phi(x)."""

    def __init__(self, th, H, noise_std=0.1, alpha=None):
        self.th, self.H, self.noise_std = th, H, noise_std
        self.D = H + 1
        self.alpha = float(H) if alpha is None else alpha
        self.m0 = np.zeros(self.D)
        self.S0 = np.eye(self.D) / self.alpha
        self.m, self.S = self.m0.copy(), self.S0.copy()

    def features(self, X):
        return phi(self.th, X, self.H)

    def posterior(self):
        return self.m.copy(), self.S.copy()

    def update(self, X, y, prior=None):
        """Exact conjugate update (this is what the PC/Newton step computes)."""
        Pm, PS = prior if prior is not None else (self.m, self.S)
        F = self.features(X)
        Pinv = np.linalg.inv(PS)
        s2 = self.noise_std ** 2
        S = np.linalg.inv(Pinv + F.T @ F / s2)
        self.m = S @ (Pinv @ Pm + F.T @ y / s2)
        self.S = 0.5 * (S + S.T)

    def predict(self, X):
        F = self.features(X)
        return F @ self.m, np.maximum(np.sum((F @ self.S) * F, axis=1), 1e-10)

    def log_evidence(self, X, y, m=None, S=None):
        m = self.m if m is None else m
        S = self.S if S is None else S
        F = self.features(X)
        C = F @ S @ F.T + self.noise_std ** 2 * np.eye(len(X))
        L = _chol(C)
        r = np.linalg.solve(L, np.asarray(y) - F @ m)
        return float(-0.5 * r @ r - np.sum(np.log(np.diag(L))) - 0.5 * len(X) * np.log(2 * np.pi))


# ------------------------------------------- continual: merge / spawn
def bll_continual(task_data, th, H, mode="evidence", noise_std=0.1, eval_data=None):
    """mode: 'merge' (one component, plain chaining), 'spawn' (one per task),
    'evidence' (merge into best component iff its evidence beats a fresh prior).
    Task-incremental evaluation. Returns (R, comps, assign, seconds)."""
    t0 = time.perf_counter()
    T = len(task_data)
    R = np.full((T, T), np.nan)
    comps, assign = [], []
    for i, (Xi, yi, _) in enumerate(task_data):
        k, merge = -1, False
        if comps:
            if mode == "merge":
                k, merge = 0, True
            elif mode == "evidence":
                ev = [c.log_evidence(Xi, yi) for c in comps]
                k = int(np.argmax(ev))
                fresh = BLL(th, H, noise_std)
                merge = ev[k] >= fresh.log_evidence(Xi, yi)
        if merge:
            comps[k].update(Xi, yi)
        else:
            c = BLL(th, H, noise_std)
            c.update(Xi, yi)
            comps.append(c)
            k = len(comps) - 1
        assign.append(k)
        for j in range(i + 1):
            Xj, fj = (eval_data[j] if eval_data is not None else (task_data[j][0], task_data[j][2]))
            mean, _ = comps[assign[j]].predict(Xj)
            R[i, j] = float(np.sqrt(np.mean((np.asarray(fj).ravel() - mean) ** 2)))
    return R, comps, assign, time.perf_counter() - t0
