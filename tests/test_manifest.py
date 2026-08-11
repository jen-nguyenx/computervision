"""Manifest: manifests/coco5k.csv shape, ordering, and determinism (D1, D3)."""

import pandas as pd

from src.data.common import build_manifest

REQUIRED_COLUMNS = ["image_id", "file_name", "width", "height", "source_split"]


def test_manifest_file_shape_and_order(cfg):
    manifest = pd.read_csv(cfg["paths"]["manifest"])

    # Exactly 5,000 rows with the D3 columns.
    assert len(manifest) == 5000
    assert list(manifest.columns) == REQUIRED_COLUMNS

    # image_id strictly ascending (implies unique) (D1).
    ids = manifest["image_id"].tolist()
    assert all(a < b for a, b in zip(ids, ids[1:]))
    assert len(set(ids)) == 5000

    # No missing values anywhere; source_split is constant.
    assert not manifest.isnull().values.any()
    assert (manifest["source_split"] == "train2017").all()


def test_build_manifest_deterministic(instances, cfg):
    n = cfg["subset"]["n_images"]
    first = build_manifest(instances, n).to_csv(index=False)
    second = build_manifest(instances, n).to_csv(index=False)

    # Same loaded JSON -> byte-identical CSV (D3 reproducibility).
    assert first == second

    # And the frozen on-disk manifest is exactly that CSV.
    on_disk = cfg["paths"]["manifest"].read_text()
    assert first == on_disk
