import numpy as np

def top_label(probs):
    return probs.max(axis=1), probs.argmax(axis=1)

def ece_score(probs, y, n_bins=15, strategy="uniform"):
    conf, pred = top_label(probs)
    correct = (pred == y).astype(float)
    if strategy == "quantile":
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        edges[-1] = 1.0 + 1e-9  # include conf == 1.0 in the top bin
    ece, N = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf >= lo) & (conf < hi)
        if msk.sum() == 0:
            continue
        ece += (msk.sum() / N) * abs(correct[msk].mean() - conf[msk].mean())
    return ece

def classwise_ece(probs, y, n_bins=10):
    """Mean one-vs-rest calibration error over classes (controls for class mix)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9  # include p == 1.0
    n = len(y)
    per_class = []
    for k in range(probs.shape[1]):
        pk, yk = probs[:, k], (y == k).astype(float)
        idx = np.digitize(pk, edges) - 1
        e = 0.0
        for m in range(n_bins):
            mask = idx == m
            if mask.any():
                e += (mask.sum() / n) * abs(yk[mask].mean() - pk[mask].mean())
        per_class.append(float(e))
    return float(np.mean(per_class)), per_class

def mce_score(probs, y, n_bins=15):
    conf, pred = top_label(probs)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    edges[-1] = 1.0 + 1e-9  # include conf == 1.0 in the top bin
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf >= lo) & (conf < hi)
        if msk.sum() > 0:
            gaps.append(abs(correct[msk].mean() - conf[msk].mean()))
    return max(gaps) if gaps else 0.0

def brier_multiclass(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

def nll_score(probs, y, eps=1e-12):
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))

def reliability_curve(probs, y, n_bins=15):
    conf, pred = top_label(probs)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    edges[-1] = 1.0 + 1e-9  # include conf == 1.0 in the top bin
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (conf >= lo) & (conf < hi)
        if msk.sum() > 0:
            xs.append(conf[msk].mean())
            ys.append(correct[msk].mean())
    return np.array(xs), np.array(ys)

def per_class_reliability(probs, y, k, n_bins=10):
    """One-vs-rest reliability curve for class k: predicted P(class=k) vs empirical rate."""
    pk = probs[:, k]
    yk = (y == k).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (pk >= lo) & (pk < hi)
        if msk.sum() > 0:
            xs.append(pk[msk].mean())
            ys.append(yk[msk].mean())
    return np.array(xs), np.array(ys)
