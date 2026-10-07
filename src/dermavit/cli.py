"""Command line interface: ``dermavit <command> [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dermavit.baseline import compute_features, run_baseline
from dermavit.config import ConfigError, Settings, load_dotenv
from dermavit.images import FolderImageSource, MemoryImageSource
from dermavit.metadata import MetadataError, load_isic2018_ground_truth, load_metadata, validate_metadata
from dermavit.metrics import seed_summary
from dermavit.reports import compare_runs, save_report
from dermavit.splits import lesion_split, lesion_overlap
from dermavit.synthetic import make_dataset, write_dataset

TEST_GT = "ISIC2018_Task3_Test_GroundTruth.csv"


def _data(args, settings: Settings) -> Path:
    return Path(args.data) if getattr(args, "data", None) else settings.data_dir


def _load_all(data: Path):
    meta = load_metadata(data / "HAM10000_metadata.csv")
    source = FolderImageSource(data / "images", data / "HAM10000_images_part_1", data / "HAM10000_images_part_2")
    test_meta = load_isic2018_ground_truth(data / TEST_GT) if (data / TEST_GT).is_file() else None
    test_source = FolderImageSource(data / "ISIC2018_Task3_Test_Images")
    return meta, source, test_meta, test_source


def _seeds(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def cmd_synth(args, settings: Settings) -> int:
    out = write_dataset(args.out, n_lesions=args.lesions, size=args.size, seed=args.seed, n_test=args.test_images)
    print(f"wrote SYNTHETIC HAM10000-style data to {out}")
    return 0


def cmd_validate(args, settings: Settings) -> int:
    meta = load_metadata(args.metadata)
    per_lesion = meta.groupby("lesion_id").size()
    print(f"OK: {len(meta)} images, {meta['lesion_id'].nunique()} lesions, "
          f"{int((per_lesion > 1).sum())} lesions with more than one image")
    print("class counts: " + json.dumps(meta["dx"].value_counts().to_dict()))
    return 0


def cmd_split(args, settings: Settings) -> int:
    meta = load_metadata(args.metadata)
    split = lesion_split(meta, args.val_fraction, args.seed if args.seed is not None else settings.seed)
    split.to_csv(args.out, index=False)
    counts = split["split"].value_counts().to_dict()
    print(f"wrote {args.out}: {counts}, shared lesions: {len(lesion_overlap(split))}")
    return 0


def cmd_baseline(args, settings: Settings) -> int:
    meta, source, test_meta, test_source = _load_all(_data(args, settings))
    ids = list(meta["image_id"])
    feats = compute_features(ids, source)
    if test_meta is not None:
        feats.update(compute_features(test_meta["image_id"], test_source))
    results = [run_baseline(meta, feats, test_meta, args.split_mode, s, target_recall=args.target_recall)
               for s in _seeds(args.seeds)]
    out = Path(args.out)
    for r in results:
        run = {"model": f"baseline_{args.split_mode}", "seed": r.seed, "val": r.val, "test": r.test,
               "test_thresholded": r.test_thresholded, "threshold": r.threshold}
        d = out / run["model"] / f"seed{r.seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.json").write_text(json.dumps(run, indent=2, default=float), encoding="utf-8")
        if r.test:
            save_report(r.test, d, "test")
    _print_summary(results)
    return 0


def _print_summary(results) -> None:
    for part in ("val", "test", "test_thresholded"):
        reps = [getattr(r, part) for r in results if getattr(r, part)]
        if not reps:
            continue
        line = []
        for m in ("balanced_accuracy", "macro_f1", "melanoma_recall", "ece"):
            s = seed_summary([rep[m] for rep in reps])
            ci = "" if np.isnan(s["ci"][0]) else f" [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
            line.append(f"{m}={s['mean']:.3f}{ci}")
        print(f"{part:16s} seeds={len(reps)} " + " ".join(line))


def cmd_train(args, settings: Settings) -> int:
    try:
        from dermavit.train import TrainConfig, train_model
    except ImportError as exc:
        raise ImportError("training needs torch. Run: pip install 'dermavit[torch]' (or [hf] for ViT and Hiera)") from exc
    meta, source, test_meta, test_source = _load_all(_data(args, settings))
    for seed in _seeds(args.seeds):
        cfg = TrainConfig(model=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                          patience=args.patience, seed=seed, weighting=args.weighting, image_size=args.image_size,
                          pretrained=not args.no_pretrained, device=settings.device, out_dir=str(args.out))
        res = train_model(cfg, lesion_split(meta, 0.2, seed), source, test_meta, test_source)
        test = res.test or {}
        print(f"{args.model} seed {seed}: best epoch {res.best_epoch} (reloaded), "
              f"val bal acc {res.val['balanced_accuracy']:.3f}, test bal acc {test.get('balanced_accuracy', float('nan')):.3f}, "
              f"melanoma recall {test.get('melanoma_recall', float('nan')):.3f}")
    return 0


def cmd_compare(args, settings: Settings) -> int:
    table = compare_runs(args.runs, args.part)
    for col in [c for c in table.columns if c.endswith("_ci")]:
        table[col] = table[col].map(lambda ci: "n/a" if ci[0] != ci[0] else f"{ci[0]:.3f}-{ci[1]:.3f}")
    print(table.round(3).to_string(index=False))
    return 0


def cmd_demo(args, settings: Settings) -> int:
    meta, images, test_meta, test_images = make_dataset(n_lesions=600, seed=0)
    meta = validate_metadata(meta)
    feats = compute_features(list(meta["image_id"]) + list(test_meta["image_id"]),
                             MemoryImageSource({**images, **test_images}))
    print(f"SYNTHETIC data: {len(meta)} images, {meta['lesion_id'].nunique()} lesions, "
          f"{len(test_meta)} external test images\n")
    for mode in ("image", "lesion"):
        print(f"split by {mode}:")
        _print_summary([run_baseline(meta, feats, test_meta, mode, s) for s in range(5)])
        print()
    print("The image split leaks lesions: its validation score is higher than its test score.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dermavit", description="Leakage-free skin-lesion benchmark (HAM10000 / ISIC 2018).")
    p.add_argument("--env-file", default=".env")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write a SYNTHETIC data folder in the HAM10000 layout")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--lesions", type=int, default=600)
    s.add_argument("--size", type=int, default=48)
    s.add_argument("--test-images", type=int, default=150)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("validate", help="check HAM10000_metadata.csv")
    s.add_argument("metadata")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("split", help="write a train/val split grouped by lesion_id")
    s.add_argument("metadata")
    s.add_argument("--out", default="split.csv")
    s.add_argument("--val-fraction", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("baseline", help="classical features + logistic regression over seeds")
    s.add_argument("--data", help="data folder (default: DERMAVIT_DATA_DIR)")
    s.add_argument("--split-mode", default="lesion", choices=["lesion", "image"])
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--target-recall", type=float, default=0.9)
    s.add_argument("--out", default="runs")
    s.set_defaults(func=cmd_baseline)

    s = sub.add_parser("train", help="train a deep model (needs the torch or hf extra)")
    s.add_argument("--data")
    s.add_argument("--model", default="vit_b16", choices=["vit_b16", "hiera_base", "efficientnet_b0", "tiny_cnn"])
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--epochs", type=int, default=10)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--patience", type=int, default=3)
    s.add_argument("--weighting", default="loss", choices=["loss", "sampler", "none"])
    s.add_argument("--image-size", type=int, default=None)
    s.add_argument("--no-pretrained", action="store_true")
    s.add_argument("--out", default="runs")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("compare", help="summarise runs/<model>/seed<k>/run.json over seeds")
    s.add_argument("--runs", default="runs")
    s.add_argument("--part", default="test", choices=["val", "test", "test_thresholded"])
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("demo", help="offline demo: image split vs lesion split on SYNTHETIC data")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file)
    try:
        return int(args.func(args, Settings.from_env()))
    except (ConfigError, MetadataError, FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
