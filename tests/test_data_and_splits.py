import pandas as pd
import pytest

from dermavit.labels import CLASSES, ISIC_COLUMNS, decode, encode, melanoma_index
from dermavit.metadata import MetadataError, load_isic2018_ground_truth, validate_metadata
from dermavit.splits import LeakageError, assert_no_lesion_overlap, image_split, lesion_overlap, lesion_split
from dermavit.synthetic import CLASS_SHARE, write_dataset


def test_label_map_is_the_single_order():
    assert encode(CLASSES) == list(range(7))
    assert decode([4]) == ["mel"] and melanoma_index() == 4
    assert set(ISIC_COLUMNS.values()) == set(CLASSES) == set(CLASS_SHARE)
    with pytest.raises(KeyError):
        encode(["melanoma"])


def test_synthetic_metadata_has_multi_image_lesions(meta):
    per_lesion = meta.groupby("lesion_id").size()
    assert (per_lesion > 1).any()
    assert meta["dx"].value_counts().idxmax() == "nv"
    assert set(meta["dx"]) == set(CLASSES)


def test_lesion_split_has_no_shared_lesion(meta):
    for seed in range(3):
        split = lesion_split(meta, 0.2, seed)
        assert lesion_overlap(split) == set()
        assert 0.1 < (split["split"] == "val").mean() < 0.3
        assert set(split.loc[split["split"] == "val", "dx"]) >= {"nv", "mel"}


def test_image_split_leaks_and_the_check_finds_it(meta):
    split = image_split(meta, 0.2, 0)
    assert lesion_overlap(split)
    with pytest.raises(LeakageError):
        assert_no_lesion_overlap(split)


def test_metadata_validation_reports_problems(meta):
    bad = meta.copy()
    bad.loc[0, "dx"] = "melanoma"
    bad.loc[1, "age"] = 300
    bad = pd.concat([bad, bad.iloc[[2]]])
    with pytest.raises(MetadataError) as info:
        validate_metadata(bad)
    text = " ".join(info.value.problems)
    assert "melanoma" in text and "age" in text and "duplicates" in text


def test_lesion_with_two_diagnoses_is_refused(meta):
    bad = meta.copy()
    lesion = bad["lesion_id"].value_counts().index[0]
    rows = bad.index[bad["lesion_id"] == lesion]
    bad.loc[rows[0], "dx"] = "mel" if bad.loc[rows[1], "dx"] != "mel" else "nv"
    with pytest.raises(MetadataError, match="more than one dx"):
        validate_metadata(bad)


def test_missing_columns(meta):
    with pytest.raises(MetadataError, match="missing columns"):
        validate_metadata(meta.drop(columns=["lesion_id"]))


def test_written_dataset_round_trip(tmp_path):
    out = write_dataset(tmp_path / "d", n_lesions=30, size=24, n_test=10)
    gt = load_isic2018_ground_truth(out / "ISIC2018_Task3_Test_GroundTruth.csv")
    assert len(gt) == 10 and set(gt["dx"]) <= set(CLASSES)
    assert len(list((out / "images").glob("*.png"))) == len(pd.read_csv(out / "HAM10000_metadata.csv"))


def test_ground_truth_needs_one_hot(tmp_path):
    path = tmp_path / "gt.csv"
    frame = pd.DataFrame({"image": ["a"], **{c: [1.0] for c in ISIC_COLUMNS}})
    frame.to_csv(path, index=False)
    with pytest.raises(MetadataError):
        load_isic2018_ground_truth(path)
