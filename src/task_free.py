"""Task-free continual learning on frozen-feature Bayesian last layers.

TRAINING without task boundaries: the stream is cut into small minibatches
(boundaries unknown); each minibatch is merged into the best component iff its
evidence beats a fresh prior, else it spawns a new component.
TEST without task identity: a few labeled CONTEXT points from the query task are
used to route (argmax evidence) or to average (Bayesian model averaging)."""
import numpy as np
from .nonlinear import BLL


def make_batches(task_data, ids, b, rng):
    """Shuffle within each task, cut into minibatches of size b. Task ids are kept
    ONLY for scoring, never shown to the learner."""
    out = []
    for (X, y, _), fid in zip(task_data, ids):
        idx = rng.permutation(len(X))
        for s in range(0, len(X), b):
            sel = idx[s:s + b]
            out.append((X[sel], y[sel], fid))
    return out


def online_bll(batches, th, H, noise_std):
    comps, counts = [], []
    for Xb, yb, fid in batches:
        k, merge = -1, False
        if comps:
            ev = [c.log_evidence(Xb, yb) for c in comps]
            k = int(np.argmax(ev))
            merge = ev[k] >= BLL(th, H, noise_std).log_evidence(Xb, yb)
        if merge:
            comps[k].update(Xb, yb)
        else:
            c = BLL(th, H, noise_std)
            c.update(Xb, yb)
            comps.append(c)
            counts.append({})
            k = len(comps) - 1
        counts[k][fid] = counts[k].get(fid, 0) + len(Xb)
    majority = [max(c, key=c.get) for c in counts]
    total = sum(sum(c.values()) for c in counts)
    purity = sum(max(c.values()) for c in counts) / total
    return comps, majority, purity


def route_scores(comps, Xc, yc):
    return np.array([c.log_evidence(Xc, yc) for c in comps])


def predict_routed(comps, scores, Xg, bma=False):
    if bma:
        w = np.exp(scores - scores.max())
        w /= w.sum()
        return sum(wk * c.predict(Xg)[0] for wk, c in zip(w, comps))
    return comps[int(np.argmax(scores))].predict(Xg)[0]
