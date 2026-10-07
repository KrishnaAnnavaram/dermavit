"""Class imbalance: class weights, sampler weights and a melanoma decision threshold."""

from __future__ import annotations

import numpy as np

from dermavit.labels import CLASSES, melanoma_index


def class_weights(y, scheme: str = "inverse", beta: float = 0.999) -> np.ndarray:
    """Weights for each class, mean 1 over the classes that occur. Absent classes get weight 0.

    ``inverse``: n / (k * count). ``effective``: (1 - beta) / (1 - beta ** count) (Cui et al., 2019).
    """
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    present = counts > 0
    w = np.zeros(len(CLASSES))
    if scheme == "inverse":
        w[present] = counts[present].sum() / (present.sum() * counts[present])
    elif scheme == "effective":
        w[present] = (1 - beta) / (1 - beta ** counts[present])
    else:
        raise ValueError(f"unknown scheme {scheme!r}: use 'inverse' or 'effective'")
    w[present] = w[present] / w[present].mean()
    return w


def sample_weights(y, scheme: str = "inverse") -> np.ndarray:
    """One weight for each sample, for a weighted (balanced) sampler."""
    return class_weights(y, scheme)[np.asarray(y, dtype=int)]


def tune_melanoma_threshold(y_val, probs_val, target_recall: float = 0.9) -> float:
    """Highest threshold on P(mel) whose melanoma recall on validation is at least ``target_recall``.

    Returns 0.5 when validation has no melanoma.
    """
    y_val = np.asarray(y_val, dtype=int)
    p_mel = np.asarray(probs_val, dtype=float)[:, melanoma_index()]
    is_mel = y_val == melanoma_index()
    if not is_mel.any():
        return 0.5
    candidates = np.unique(np.concatenate([p_mel[is_mel], [0.0]]))[::-1]
    for t in candidates:
        if (p_mel[is_mel] >= t).mean() >= target_recall:
            return float(t)
    return 0.0


def apply_melanoma_threshold(probs, threshold: float) -> np.ndarray:
    """Predict melanoma when P(mel) >= threshold. Otherwise predict the most probable other class."""
    p = np.asarray(probs, dtype=float)
    k = melanoma_index()
    others = p.copy()
    others[:, k] = -1.0
    pred = others.argmax(1)
    pred[p[:, k] >= threshold] = k
    return pred
