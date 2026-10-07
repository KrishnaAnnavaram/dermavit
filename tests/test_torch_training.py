"""Tests of the deep-learning path. They skip when torch is not installed (as in CI)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dermavit.explain import grad_cam  # noqa: E402
from dermavit.models import TinyCNN, build_model, count_parameters  # noqa: E402
from dermavit.splits import lesion_split  # noqa: E402
from dermavit.train import TrainConfig, train_model  # noqa: E402


@pytest.fixture(scope="module")
def run(synth, source, tmp_path_factory):
    meta, _, test_meta, _ = synth
    cfg = TrainConfig(model="tiny_cnn", epochs=3, batch_size=64, lr=3e-3, patience=1, seed=0, image_size=32,
                      pretrained=False, device="cpu", out_dir=str(tmp_path_factory.mktemp("runs")))
    return cfg, train_model(cfg, lesion_split(meta, 0.2, 0), source, test_meta, source)


def test_tiny_cnn_shapes():
    model = TinyCNN()
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 7)
    assert count_parameters(model) > 1000
    assert isinstance(build_model("tiny_cnn"), TinyCNN)


def test_training_reloads_the_best_checkpoint(run):
    cfg, res = run
    assert res.reloaded_best and 0 <= res.best_epoch < len(res.history)
    best = max(h.val_balanced_accuracy for h in res.history)
    assert res.val["balanced_accuracy"] == pytest.approx(best, abs=1e-9)


def test_every_epoch_is_logged_with_time(run):
    _, res = run
    assert all(h.epoch_seconds >= h.train_seconds > 0 for h in res.history)
    assert res.efficiency["epochs_run"] == len(res.history) and res.efficiency["peak_memory_mb"] is None  # CPU


def test_run_json_is_written(run):
    cfg, res = run
    path = __import__("pathlib").Path(cfg.out_dir) / "tiny_cnn" / "seed0" / "run.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["test"]["confusion"]["labels"][4] == "mel" and data["reloaded_best"] is True


def test_training_is_reproducible(synth, source, tmp_path):
    meta, _, test_meta, _ = synth
    split = lesion_split(meta, 0.2, 0)
    outs = []
    for k in range(2):
        cfg = TrainConfig(model="tiny_cnn", epochs=1, batch_size=64, seed=7, image_size=32, pretrained=False,
                          device="cpu", out_dir=str(tmp_path / f"r{k}"))
        outs.append(train_model(cfg, split, source).history[0].train_loss)
    assert outs[0] == pytest.approx(outs[1], rel=1e-6)


def test_grad_cam_map():
    model = TinyCNN()
    cam = grad_cam(model, torch.rand(1, 3, 32, 32))
    assert cam.shape == (32, 32) and np.nanmax(cam) <= 1.0 + 1e-6
    with pytest.raises(ValueError):
        grad_cam(model, torch.rand(2, 3, 32, 32))
