"""Splits: manifests/coco5k_splits.csv partitions and seeded reproducibility (D4)."""

import pandas as pd

from src.data.make_splits import assign_splits

PARTITIONS = ["train", "val", "test"]
EXPECTED_SIZES = {"train": 4000, "val": 500, "test": 500}


def test_splits_partition_the_manifest(cfg):
    manifest = pd.read_csv(cfg["paths"]["manifest"])
    splits = pd.read_csv(cfg["paths"]["splits"])

    assert list(splits.columns) == ["image_id", "split"]
    assert set(splits["split"].unique()) == set(PARTITIONS)

    # Pairwise disjoint: each image_id appears exactly once.
    assert splits["image_id"].is_unique

    # Union == manifest ids.
    assert set(splits["image_id"]) == set(manifest["image_id"])

    # Sizes exactly 4000/500/500 (D4).
    sizes = splits["split"].value_counts().to_dict()
    assert sizes == EXPECTED_SIZES


def test_split_algorithm_reproduces_frozen_file(cfg):
    manifest = pd.read_csv(cfg["paths"]["manifest"])
    splits = pd.read_csv(cfg["paths"]["splits"])
    frozen = dict(zip(splits["image_id"], splits["split"]))

    # Re-run the seeded algorithm (seed 0 from the config) on the manifest ids.
    rebuilt = assign_splits(manifest["image_id"].tolist(), cfg["seed"], cfg["split"]["ratios"])

    assert cfg["seed"] == 0  # the frozen split was made with seed 0 (D4)
    assert rebuilt == frozen
