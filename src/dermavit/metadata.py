"""Load and validate the HAM10000 metadata and the ISIC 2018 Task 3 test ground truth."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dermavit.labels import CLASSES, ISIC_COLUMNS

META_COLUMNS = ("lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization")
DX_TYPES = ("histo", "follow_up", "consensus", "confocal")
SEXES = ("male", "female", "unknown")


class MetadataError(ValueError):
    """The metadata breaks the contract. ``problems`` lists each failed rule."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("metadata validation failed:\n  - " + "\n  - ".join(problems))


def validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy or raise MetadataError with every problem found."""
    missing = [c for c in META_COLUMNS if c not in df.columns]
    if missing:
        raise MetadataError([f"missing columns: {missing}"])
    out = df[list(META_COLUMNS)].copy()
    problems: list[str] = []
    for col in ("lesion_id", "image_id", "dx"):
        if out[col].isna().any():
            problems.append(f"{col}: {int(out[col].isna().sum())} missing values")
    out["image_id"] = out["image_id"].astype(str).str.strip()
    out["lesion_id"] = out["lesion_id"].astype(str).str.strip()
    out["dx"] = out["dx"].astype(str).str.strip().str.lower()
    bad_dx = sorted(set(out["dx"]) - set(CLASSES))
    if bad_dx:
        problems.append(f"dx: unknown classes {bad_dx}")
    bad_type = sorted(set(out["dx_type"].dropna().astype(str)) - set(DX_TYPES))
    if bad_type:
        problems.append(f"dx_type: unknown values {bad_type}")
    out["sex"] = out["sex"].fillna("unknown").astype(str).str.lower()
    bad_sex = sorted(set(out["sex"]) - set(SEXES))
    if bad_sex:
        problems.append(f"sex: unknown values {bad_sex}")
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    if ((out["age"] < 0) | (out["age"] > 120)).any():
        problems.append("age: values outside 0-120")
    dup = out["image_id"].duplicated()
    if dup.any():
        problems.append(f"image_id: {int(dup.sum())} duplicates")
    per_lesion = out.groupby("lesion_id")["dx"].nunique()
    mixed = per_lesion[per_lesion > 1]
    if len(mixed):
        problems.append(f"{len(mixed)} lesions have more than one dx, first {mixed.index[0]}")
    if problems:
        raise MetadataError(problems)
    return out.reset_index(drop=True)


def load_metadata(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} does not exist. See data/README.md, or run `dermavit synth`.")
    return validate_metadata(pd.read_csv(p))


def load_isic2018_ground_truth(path: str | Path) -> pd.DataFrame:
    """Change the one-hot ISIC 2018 Task 3 ground truth into ``image_id, dx``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} does not exist. See data/README.md.")
    raw = pd.read_csv(p)
    cols = [c for c in ISIC_COLUMNS if c in raw.columns]
    if "image" not in raw.columns or len(cols) != len(ISIC_COLUMNS):
        raise MetadataError([f"expected columns image + {list(ISIC_COLUMNS)}"])
    onehot = raw[list(ISIC_COLUMNS)].to_numpy()
    if not ((onehot.sum(axis=1) == 1).all()):
        raise MetadataError(["each ground-truth row must have exactly one 1"])
    dx = [ISIC_COLUMNS[list(ISIC_COLUMNS)[i]] for i in onehot.argmax(axis=1)]
    return pd.DataFrame({"image_id": raw["image"].astype(str), "dx": dx})
