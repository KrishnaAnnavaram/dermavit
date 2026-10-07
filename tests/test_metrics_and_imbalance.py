import math

import numpy as np
import pytest

from dermavit.imbalance import apply_melanoma_threshold, class_weights, sample_weights, tune_melanoma_threshold
from dermavit.labels import CLASSES, INDEX
from dermavit.metrics import (check_probs, classification_report, expected_calibration_error, lesion_bootstrap_ci,
                              ovr_auc, seed_summary)


def onehot(idx, n=7, p=0.9):
    out = np.full((len(idx), n), (1 - p) / (n - 1))
    out[np.arange(len(idx)), idx] = p
    return out


def test_always_nevus_has_high_accuracy_but_low_balanced_accuracy():
    y = np.array([INDEX["nv"]] * 67 + [INDEX["mel"]] * 11 + [INDEX["bkl"]] * 11 + [INDEX["bcc"]] * 11)
    rep = classification_report(y, onehot(np.full(len(y), INDEX["nv"])))
    assert rep["accuracy"] == pytest.approx(0.67)
    assert rep["balanced_accuracy"] == pytest.approx(0.25)
    assert rep["melanoma_recall"] == 0.0


def test_confusion_matrix_uses_the_label_order():
    y = np.array([INDEX["mel"], INDEX["nv"]])
    rep = classification_report(y, onehot(np.array([INDEX["mel"], INDEX["mel"]])))
    assert rep["confusion"]["labels"] == list(CLASSES)
    m = np.array(rep["confusion"]["matrix"])
    assert m[INDEX["mel"], INDEX["mel"]] == 1 and m[INDEX["nv"], INDEX["mel"]] == 1


def test_auc_comes_from_probabilities_not_hard_labels():
    y = np.array([0, 0, 1, 1])
    p = np.zeros((4, 7))
    p[:, 0] = [0.9, 0.6, 0.4, 0.2]
    p[:, 1] = 1 - p[:, 0]
    auc = ovr_auc(y, p)
    assert auc["akiec"] == 1.0 and set(auc) == {"akiec", "bcc", "macro"}


def test_probability_shape_is_checked():
    with pytest.raises(ValueError):
        check_probs(np.ones((3, 5)) / 5)
    with pytest.raises(ValueError):
        check_probs(np.ones((3, 7)))


def test_ece_is_zero_for_perfectly_calibrated_certainty():
    y = np.array([0, 1, 2])
    assert expected_calibration_error(y, onehot(y, p=1.0)) == pytest.approx(0.0)
    assert expected_calibration_error(y, onehot(np.array([1, 2, 0]), p=1.0)) == pytest.approx(1.0)


def test_lesion_bootstrap_interval():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    pred = np.where(rng.random(200) < 0.8, y, (y + 1) % 3)
    lo, hi = lesion_bootstrap_ci(y, pred, np.arange(200) // 2, n_boot=200)
    assert lo < 0.8 < hi


def test_seed_summary_t_interval():
    s = seed_summary([0.70, 0.72, 0.74])
    assert s["mean"] == pytest.approx(0.72) and s["ci"][0] < 0.70 and s["ci"][1] > 0.74
    assert math.isnan(seed_summary([0.5])["sd"])


def test_class_weights_favour_rare_classes():
    y = np.array([INDEX["nv"]] * 90 + [INDEX["mel"]] * 10)
    w = class_weights(y)
    assert w[INDEX["mel"]] > w[INDEX["nv"]] and w[INDEX["df"]] == 0
    assert w[w > 0].mean() == pytest.approx(1.0)
    sw = sample_weights(y)
    assert sw[y == INDEX["mel"]].sum() == pytest.approx(sw[y == INDEX["nv"]].sum())
    assert class_weights(y, "effective")[INDEX["mel"]] > class_weights(y, "effective")[INDEX["nv"]]
    with pytest.raises(ValueError):
        class_weights(y, "magic")


def test_melanoma_threshold_reaches_target_recall():
    mel, nv = INDEX["mel"], INDEX["nv"]
    y = np.array([mel] * 10 + [nv] * 10)
    p = np.zeros((20, 7))
    p[:, mel] = np.r_[np.linspace(0.1, 0.9, 10), np.linspace(0.0, 0.3, 10)]
    p[:, nv] = 1 - p[:, mel]
    t = tune_melanoma_threshold(y, p, target_recall=0.9)
    pred = apply_melanoma_threshold(p, t)
    assert (pred[:10] == mel).mean() >= 0.9
    assert tune_melanoma_threshold(np.array([nv]), p[:1]) == 0.5
