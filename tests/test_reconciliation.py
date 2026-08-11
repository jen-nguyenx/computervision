"""Conversion ledgers: the D11 reconciliation identity, recomputed from the raw
counts (never trusting the stored reconciliation_ok flag), plus label-file
counts per task and per split (D1, D4, D14).
"""

import json

import pytest

from src.data.common import read_manifest

TASKS = ["detect", "segment"]
EXPECTED_SPLIT_SIZES = {"train": 4000, "val": 500, "test": 500}


@pytest.fixture(scope="session")
def subset_annotations(instances, cfg):
    """Annotations restricted to the manifest images, recounted from the raw json
    — the independent ground truth the ledgers must agree with."""
    manifest_ids = set(read_manifest(cfg)["image_id"])
    return [ann for ann in instances["annotations"] if ann["image_id"] in manifest_ids]


@pytest.fixture(scope="session")
def ledgers(cfg):
    loaded = {}
    for task in TASKS:
        with open(cfg["paths"]["metrics_dir"] / f"conversion_{task}.json") as f:
            loaded[task] = json.load(f)
    return loaded


@pytest.mark.parametrize("task", TASKS)
def test_reconciliation_identity_recomputed(ledgers, subset_annotations, task):
    ledger = ledgers[task]
    excluded = ledger["excluded"]

    # The stored flag must be true...
    assert ledger["reconciliation_ok"] is True

    # input_annotations recounted INDEPENDENTLY from the raw json — if the
    # converter under-built its annotation index, input and converted would
    # shrink together and the identity below would still balance; this catches it.
    assert ledger["input_annotations"] == len(subset_annotations)

    # ...but we recompute the identity ourselves (D11). degenerate_parts counts
    # dropped polygon PARTS, not whole annotations, so it sits outside the
    # whole-annotation identity; here it is 0 (expected), so the naive
    # "converted + sum of all excluded values" identity holds too.
    whole_annotation_exclusions = (
        excluded["crowd"] + excluded["zero_area"] + excluded["empty_segmentation"]
    )
    assert ledger["input_annotations"] == ledger["converted"] + whole_annotation_exclusions
    assert excluded["degenerate_parts"] == 0
    assert ledger["input_annotations"] == ledger["converted"] + sum(excluded.values())


@pytest.mark.parametrize("task", TASKS)
def test_ledger_expected_numbers(ledgers, task):
    ledger = ledgers[task]
    assert ledger["converted"] == 35765
    assert ledger["excluded"]["crowd"] == 448


@pytest.mark.parametrize("task", TASKS)
def test_label_file_counts(label_files, task):
    per_split = {split: len(paths) for split, paths in label_files[task].items()}
    assert per_split == EXPECTED_SPLIT_SIZES  # 4000/500/500 (D4)
    assert sum(per_split.values()) == 5000  # one label file per image (D1)


def keepable(ann, task):
    """Would the converter keep this annotation, per the task's policy?

    Detect keeps non-crowd with a positive-area bbox (D6, D12); segment keeps
    non-crowd with >= 1 valid polygon part (D6, D8). Mirrors the documented
    policy, recomputed from the raw json rather than from converter internals.
    """
    if ann.get("iscrowd", 0) == 1:
        return False
    if task == "detect":
        return ann["bbox"][2] > 0 and ann["bbox"][3] > 0
    seg = ann["segmentation"]
    return isinstance(seg, list) and any(len(p) >= 6 and len(p) % 2 == 0 for p in seg)


@pytest.mark.parametrize("task", TASKS)
def test_empty_label_files_exact(label_files, subset_annotations, cfg, task):
    # An image's label file is empty exactly when NONE of its annotations are
    # keepable under the task policy (truly-empty images included, D1). Exact
    # equality also catches a converter bug that silently emptied extra files.
    # (49 in this subset — no crowd-only images here — recomputed, not hard-coded.)
    images_with_keepable = {
        ann["image_id"] for ann in subset_annotations if keepable(ann, task)
    }
    expected_empty = len(set(read_manifest(cfg)["image_id"]) - images_with_keepable)
    empty = sum(
        1 for paths in label_files[task].values() for path in paths if path.stat().st_size == 0
    )
    assert empty == expected_empty
