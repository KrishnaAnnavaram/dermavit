"""Train / validation split grouped by ``lesion_id``.

HAM10000 has several images of the same lesion. An image-level split puts near-duplicate images in
train and validation, so the validation score is too high. ``lesion_split`` keeps each lesion in one
part and stratifies by diagnosis. ``image_split`` exists only to measure that leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class LeakageError(AssertionError):
    """A lesion appears in more than one split."""


def lesion_split(meta: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Return ``meta`` with a ``split`` column (``train`` / ``val``), grouped by lesion."""
    if not 0.05 <= val_fraction <= 0.5:
        raise ValueError("val_fraction must be between 0.05 and 0.5")
    n_splits = max(2, int(round(1 / val_fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(meta, meta["dx"], groups=meta["lesion_id"]))
    out = meta.copy()
    out["split"] = "train"
    out.loc[out.index[val_idx], "split"] = "val"
    assert_no_lesion_overlap(out)
    return out


def image_split(meta: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Image-level stratified split. It leaks lesions. Use it only for the leakage comparison."""
    _, val_idx = train_test_split(np.arange(len(meta)), test_size=val_fraction, stratify=meta["dx"],
                                  random_state=seed)
    out = meta.copy()
    out["split"] = "train"
    out.loc[out.index[val_idx], "split"] = "val"
    return out


def lesion_overlap(df: pd.DataFrame) -> set[str]:
    parts = [set(df.loc[df["split"] == s, "lesion_id"]) for s in sorted(df["split"].unique())]
    shared: set[str] = set()
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            shared |= parts[i] & parts[j]
    return shared


def assert_no_lesion_overlap(df: pd.DataFrame) -> None:
    shared = lesion_overlap(df)
    if shared:
        raise LeakageError(f"{len(shared)} lesions are in more than one split, first {sorted(shared)[0]}")
