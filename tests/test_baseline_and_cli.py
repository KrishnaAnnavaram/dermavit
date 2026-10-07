import json

import numpy as np
import pytest

from dermavit.baseline import compute_features, full_probs, run_baseline
from dermavit.cli import main
from dermavit.config import ConfigError, Settings
from dermavit.reports import compare_runs, save_report


@pytest.fixture(scope="module")
def feats(synth, source):
    meta, _, test_meta, _ = synth
    return compute_features(list(meta["image_id"]) + list(test_meta["image_id"]), source)


def test_baseline_reports_all_parts(synth, feats):
    meta, _, test_meta, _ = synth
    r = run_baseline(meta, feats, test_meta, "lesion", seed=0)
    for rep in (r.val, r.test, r.test_thresholded):
        assert 0 <= rep["balanced_accuracy"] <= 1 and "mel" in rep["recall"]
    assert "balanced_accuracy_ci" in r.val
    assert 0 <= r.threshold <= 1


def test_baseline_beats_chance(synth, feats):
    meta, _, test_meta, _ = synth
    r = run_baseline(meta, feats, test_meta, "lesion", seed=1)
    assert r.test["balanced_accuracy"] > 1 / 7 + 0.15


def test_image_split_inflates_validation(synth, feats):
    meta, _, test_meta, _ = synth
    gaps = {}
    for mode in ("lesion", "image"):
        rs = [run_baseline(meta, feats, test_meta, mode, seed=s) for s in range(3)]
        gaps[mode] = np.mean([r.val["balanced_accuracy"] - r.test["balanced_accuracy"] for r in rs])
    assert gaps["image"] > gaps["lesion"]


def test_full_probs_fills_absent_classes(synth, feats):
    from dermavit.baseline import fit_baseline
    meta = synth[0]
    keep = meta[meta["dx"].isin(["nv", "mel"])]
    X = np.stack([feats[i] for i in keep["image_id"]])
    y = np.where(keep["dx"] == "mel", 4, 5)
    p = full_probs(fit_baseline(X, y), X)
    assert p.shape[1] == 7 and np.allclose(p[:, [0, 1, 2, 3, 6]], 0)


def test_settings():
    assert Settings.from_env({}).seed == 42
    with pytest.raises(ConfigError):
        Settings.from_env({"DERMAVIT_DEVICE": "tpu"})
    with pytest.raises(ConfigError):
        Settings.from_env({"DERMAVIT_SEED": "x"})


def test_cli_round_trip(tmp_path, capsys):
    data = tmp_path / "data"
    assert main(["synth", "--out", str(data), "--lesions", "120", "--size", "24", "--test-images", "30"]) == 0
    assert main(["validate", str(data / "HAM10000_metadata.csv")]) == 0
    assert main(["split", str(data / "HAM10000_metadata.csv"), "--out", str(tmp_path / "s.csv")]) == 0
    assert "shared lesions: 0" in capsys.readouterr().out
    runs = tmp_path / "runs"
    assert main(["baseline", "--data", str(data), "--seeds", "0,1", "--out", str(runs)]) == 0
    assert (runs / "baseline_lesion" / "seed1" / "test_confusion.csv").is_file()
    capsys.readouterr()
    assert main(["compare", "--runs", str(runs)]) == 0
    assert "baseline_lesion" in capsys.readouterr().out


def test_compare_and_save_report(tmp_path):
    rep = {"balanced_accuracy": 0.5, "macro_f1": 0.4, "melanoma_recall": 0.6, "ece": 0.1, "auc_ovr": {"macro": 0.8},
           "confusion": {"labels": ["a", "b"], "matrix": [[1, 0], [0, 1]]}}
    for seed, ba in ((0, 0.5), (1, 0.6)):
        d = tmp_path / "m" / f"seed{seed}"
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps({"model": "m", "seed": seed, "test": {**rep, "balanced_accuracy": ba}}))
    table = compare_runs(tmp_path)
    assert table.loc[0, "seeds"] == 2 and table.loc[0, "balanced_accuracy"] == pytest.approx(0.55)
    path = save_report(rep, tmp_path / "out")
    assert path.is_file() and (tmp_path / "out" / "report_confusion.csv").read_text().startswith(",pred_a")


def test_cli_errors(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "missing.csv")]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["compare", "--runs", str(tmp_path)]) == 1


def test_demo(capsys):
    assert main(["demo"]) == 0
    assert "split by lesion" in capsys.readouterr().out
