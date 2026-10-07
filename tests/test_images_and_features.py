import numpy as np
import pytest
from PIL import Image

from dermavit.features import FEATURE_NAMES, image_features
from dermavit.images import (IMAGENET_MEAN, MODEL_SPECS, EvalTransform, FolderImageSource, TrainTransform,
                             normalise, spec_for, transforms_for)


def test_eval_transform_is_deterministic(source, meta):
    img = source.get(meta["image_id"].iloc[0])
    _, ev = transforms_for("hiera_base", 32)
    a, b = ev(img), ev(img, np.random.default_rng(1))
    assert np.array_equal(a, b) and a.shape == (3, 32, 32) and a.dtype == np.float32


def test_train_transform_is_random_but_seeded(source, meta):
    img = source.get(meta["image_id"].iloc[0])
    tr, _ = transforms_for("tiny_cnn", 32)
    a = tr(img, np.random.default_rng(5))
    b = tr(img, np.random.default_rng(5))
    c = tr(img, np.random.default_rng(6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_each_model_uses_its_own_normalisation():
    assert spec_for("vit_b16").mean == (0.5, 0.5, 0.5)
    assert spec_for("hiera_base").mean == IMAGENET_MEAN
    assert spec_for("efficientnet_b0").std == (0.229, 0.224, 0.225)
    _, ev_vit = transforms_for("vit_b16")
    _, ev_hiera = transforms_for("hiera_base")
    img = np.full((8, 8, 3), 128, np.uint8)
    assert not np.allclose(ev_vit(img), ev_hiera(img))
    with pytest.raises(KeyError):
        spec_for("resnet9000")


def test_normalise_values():
    img = np.zeros((2, 2, 3), np.uint8)
    out = normalise(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    assert np.allclose(out, -1.0)


def test_folder_source_reads_jpg_and_png(tmp_path):
    Image.fromarray(np.full((5, 5, 3), 200, np.uint8)).save(tmp_path / "ISIC_0000001.png")
    (tmp_path / "sub").mkdir()
    Image.fromarray(np.full((5, 5, 3), 10, np.uint8)).save(tmp_path / "sub" / "ISIC_0000002.jpg")
    src = FolderImageSource(tmp_path)
    assert src.get("ISIC_0000001").shape == (5, 5, 3)
    assert src.get("ISIC_0000002").mean() < 30
    with pytest.raises(FileNotFoundError):
        src.get("ISIC_9999999")


def test_features_have_fixed_length_and_names(source, meta):
    f = image_features(source.get(meta["image_id"].iloc[0]))
    assert f.shape == (len(FEATURE_NAMES),) and np.isfinite(f).all()


def test_all_model_specs_have_three_channel_stats():
    for spec in MODEL_SPECS.values():
        assert len(spec.mean) == len(spec.std) == 3
    assert isinstance(TrainTransform(8, (0, 0, 0), (1, 1, 1)), TrainTransform)
    assert isinstance(EvalTransform(8, (0, 0, 0), (1, 1, 1)), EvalTransform)
