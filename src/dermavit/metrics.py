"""Metrics for an imbalanced 7-class medical task.

The headline metric is balanced accuracy (the ISIC 2018 Task 3 metric). The report also gives macro
F1, the recall of each class (melanoma first), one-vs-rest ROC-AUC from probabilities, the expected
calibration error, and a bootstrap interval that resamples lesions, not images.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from dermavit.labels import CLASSES, MELANOMA


def check_probs(probs: np.ndarray, n_classes: int = len(CLASSES)) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError(f"probabilities must have shape (n, {n_classes}), got {p.shape}")
    if not np.allclose(p.sum(1), 1.0, atol=1e-3) or (p < 0).any():
        raise ValueError("each probability row must be non-negative and sum to 1")
    return p


def ovr_auc(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """One-vs-rest ROC-AUC for each class that has positives and negatives, plus the macro mean."""
    p = check_probs(probs)
    out = {}
    for k, name in enumerate(CLASSES):
        pos = y_true == k
        if pos.any() and (~pos).any():
            out[name] = float(roc_auc_score(pos.astype(int), p[:, k]))
    out["macro"] = float(np.mean(list(out.values()))) if out else float("nan")
    return out


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    p = check_probs(probs)
    conf = p.max(1)
    correct = (p.argmax(1) == y_true).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def lesion_bootstrap_ci(y_true, y_pred, groups, n_boot: int = 500, seed: int = 0, level: float = 0.95):
    """Percentile interval of balanced accuracy. Each resample draws lesions with replacement."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    with warnings.catch_warnings():  # a resample can miss a predicted class
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_boot):
            idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
            if len(np.unique(y_true[idx])) < 2:
                continue
            stats.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def classification_report(y_true, probs, y_pred=None, groups=None, seed: int = 0) -> dict:
    """All metrics for one prediction set. ``y_pred`` defaults to the argmax of ``probs``."""
    y_true = np.asarray(y_true, dtype=int)
    p = check_probs(probs)
    y_pred = p.argmax(1) if y_pred is None else np.asarray(y_pred, dtype=int)
    labels = list(range(len(CLASSES)))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    present = set(np.unique(y_true))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    report = {
        "n": int(len(y_true)),
        "balanced_accuracy": bal_acc,
        "macro_f1": float(f1_score(y_true, y_pred, labels=sorted(present), average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "recall": {CLASSES[k]: float(recalls[k]) for k in labels if k in present},
        "melanoma_recall": float(recalls[CLASSES.index(MELANOMA)]) if CLASSES.index(MELANOMA) in present else float("nan"),
        "auc_ovr": ovr_auc(y_true, p),
        "ece": expected_calibration_error(y_true, p),
        "confusion": {"labels": list(CLASSES), "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }
    if groups is not None:
        report["balanced_accuracy_ci"] = lesion_bootstrap_ci(y_true, y_pred, groups, seed=seed)
    return report


def seed_summary(values: list[float], level: float = 0.95) -> dict:
    """Mean, standard deviation and a t-interval over seeds."""
    from scipy import stats

    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return {"n": n, "mean": mean, "sd": float("nan"), "ci": (float("nan"), float("nan"))}
    sd = float(v.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci": (mean - half, mean + half)}
