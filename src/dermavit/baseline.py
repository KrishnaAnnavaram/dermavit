"""Classical baseline: handcrafted features + class-weighted logistic regression.

It uses the same split, the same metrics, the same melanoma threshold procedure and the same
external test set as the transformer models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dermavit.features import feature_matrix
from dermavit.imbalance import apply_melanoma_threshold, tune_melanoma_threshold
from dermavit.labels import CLASSES, encode
from dermavit.metrics import classification_report
from dermavit.splits import image_split, lesion_split


@dataclass
class BaselineResult:
    split_mode: str
    seed: int
    val: dict
    test: dict | None
    threshold: float
    test_thresholded: dict | None


def full_probs(model, X: np.ndarray) -> np.ndarray:
    """predict_proba with one column for each of the 7 classes (0 for a class absent in training)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASSES)))
    out[:, model.classes_] = p
    return out


def fit_baseline(X: np.ndarray, y: np.ndarray, seed: int = 0, balanced: bool = True):
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced" if balanced else None, random_state=seed)
    return make_pipeline(StandardScaler(), clf).fit(X, y)


def run_baseline(meta: pd.DataFrame, features: dict[str, np.ndarray], test_meta: pd.DataFrame | None = None,
                 split_mode: str = "lesion", seed: int = 0, val_fraction: float = 0.2,
                 target_recall: float = 0.9, balanced: bool = True) -> BaselineResult:
    """``features`` maps image id -> feature vector (see ``features.feature_matrix``)."""
    if split_mode == "lesion":
        split = lesion_split(meta, val_fraction, seed)
    elif split_mode == "image":
        split = image_split(meta, val_fraction, seed)
    else:
        raise ValueError("split_mode must be 'lesion' or 'image'")
    tr, va = split[split["split"] == "train"], split[split["split"] == "val"]
    X_tr = np.stack([features[i] for i in tr["image_id"]])
    X_va = np.stack([features[i] for i in va["image_id"]])
    y_tr, y_va = np.asarray(encode(tr["dx"])), np.asarray(encode(va["dx"]))
    model = fit_baseline(X_tr, y_tr, seed, balanced)
    p_va = full_probs(model, X_va)
    val = classification_report(y_va, p_va, groups=va["lesion_id"].to_numpy(), seed=seed)
    threshold = tune_melanoma_threshold(y_va, p_va, target_recall)
    test = test_thr = None
    if test_meta is not None and len(test_meta):
        X_te = np.stack([features[i] for i in test_meta["image_id"]])
        y_te = np.asarray(encode(test_meta["dx"]))
        p_te = full_probs(model, X_te)
        test = classification_report(y_te, p_te, seed=seed)
        test_thr = classification_report(y_te, p_te, y_pred=apply_melanoma_threshold(p_te, threshold), seed=seed)
    return BaselineResult(split_mode, seed, val, test, threshold, test_thr)


def compute_features(image_ids, source) -> dict[str, np.ndarray]:
    ids = list(image_ids)
    return dict(zip(ids, feature_matrix(ids, source)))
