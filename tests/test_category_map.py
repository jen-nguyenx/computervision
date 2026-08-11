"""Category map: build_category_map on the real COCO categories (D2).

Original ids run 1-90 with gaps; the map must be an explicit sorted-original-id
-> 0..79 assignment, never `id - 1`.
"""

import json

from src.data.common import build_category_map


def test_category_map_on_real_categories(instances):
    cat_map = build_category_map(instances["categories"])

    # Exactly 80 classes.
    assert len(cat_map["map"]) == 80
    assert len(cat_map["names"]) == 80

    # Contiguous 0..79, no gaps, no duplicates.
    assert sorted(cat_map["map"].values()) == list(range(80))
    assert sorted(int(k) for k in cat_map["names"]) == list(range(80))

    # person is original id 1, the lowest, so it maps to 0.
    assert cat_map["map"]["1"] == 0
    assert cat_map["names"]["0"] == "person"

    # Sorted-original-id order: contiguous ids follow ascending original ids.
    originals_in_contiguous_order = sorted(cat_map["map"], key=lambda k: cat_map["map"][k])
    assert [int(k) for k in originals_in_contiguous_order] == sorted(
        int(k) for k in cat_map["map"]
    )


def test_category_map_matches_saved_file(instances, cfg):
    # The frozen configs/coco_category_map.json must equal a fresh rebuild (D2, D3).
    with open(cfg["paths"]["category_map"]) as f:
        saved = json.load(f)
    assert saved == build_category_map(instances["categories"])
