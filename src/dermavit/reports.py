"""Save run reports and compare models over seeds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dermavit.metrics import seed_summary


def save_report(report: dict, out_dir: str | Path, name: str = "report") -> Path:
    """Write ``<name>.json`` and ``<name>_confusion.csv`` (rows: true class, columns: predicted class)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    conf = report.get("confusion")
    if conf:
        frame = pd.DataFrame(conf["matrix"], index=[f"true_{c}" for c in conf["labels"]],
                             columns=[f"pred_{c}" for c in conf["labels"]])
        frame.to_csv(out / f"{name}_confusion.csv")
    return out / f"{name}.json"


METRICS = ("balanced_accuracy", "macro_f1", "melanoma_recall", "ece")


def compare_runs(runs_dir: str | Path, part: str = "test") -> pd.DataFrame:
    """Read ``<runs_dir>/<model>/seed<k>/run.json`` files and summarise each model over seeds."""
    rows = []
    for path in sorted(Path(runs_dir).glob("*/seed*/run.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        rep = run.get(part)
        if rep:
            rows.append({"model": run["model"], "seed": run["seed"], **{m: rep[m] for m in METRICS},
                         "auc_macro": rep["auc_ovr"]["macro"],
                         "peak_memory_mb": run.get("efficiency", {}).get("peak_memory_mb"),
                         "train_images_per_s": run.get("efficiency", {}).get("train_images_per_s")})
    if not rows:
        raise FileNotFoundError(f"no */seed*/run.json with a {part!r} part under {runs_dir}")
    frame = pd.DataFrame(rows)
    out = []
    for model, part_df in frame.groupby("model"):
        row = {"model": model, "seeds": len(part_df)}
        for m in (*METRICS, "auc_macro"):
            s = seed_summary(part_df[m].astype(float).tolist())
            row[m] = s["mean"]
            row[f"{m}_ci"] = s["ci"]
        row["peak_memory_mb"] = part_df["peak_memory_mb"].dropna().mean() if part_df["peak_memory_mb"].notna().any() else None
        row["train_images_per_s"] = part_df["train_images_per_s"].dropna().mean() if part_df["train_images_per_s"].notna().any() else None
        out.append(row)
    return pd.DataFrame(out)
