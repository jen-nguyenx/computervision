"""Written label files: sample >= 500 per task and check every line (D5, D10, D12).

Detect lines are `cls cx cy w h` (5 fields, w/h > 0); segment lines are
`cls x1 y1 x2 y2 ...` (odd field count >= 7, i.e. class + at least 3 points).
All coordinates are normalized into [0, 1] and class ids are contiguous 0..79.
"""

import os
import random
from pathlib import Path

import pytest

SAMPLE_SIZE = 600  # deterministic sample, comfortably above the 500 floor


def sample_label_paths(label_files, task):
    all_paths = [path for paths in label_files[task].values() for path in paths]
    assert len(all_paths) == 5000
    return random.Random(0).sample(sorted(all_paths), SAMPLE_SIZE)


def check_common_fields(fields, path):
    """Class id in [0, 79] and every coordinate in [0, 1] (D2, D10)."""
    cls = int(fields[0])
    assert fields[0] == str(cls), f"non-integer class in {path}"
    assert 0 <= cls <= 79, f"class {cls} out of range in {path}"
    for value in fields[1:]:
        coord = float(value)
        assert 0.0 <= coord <= 1.0, f"coordinate {coord} out of [0,1] in {path}"


def test_detect_labels_sample(label_files):
    lines_checked = 0
    for path in sample_label_paths(label_files, "detect"):
        for line in path.read_text().splitlines():
            fields = line.split()
            assert len(fields) == 5, f"detect line with {len(fields)} fields in {path}"
            check_common_fields(fields, path)
            w, h = float(fields[3]), float(fields[4])
            assert w > 0 and h > 0, f"zero-area box in {path}"  # D12: those were dropped
            lines_checked += 1
    assert lines_checked > 0


def test_segment_labels_sample(label_files):
    lines_checked = 0
    for path in sample_label_paths(label_files, "segment"):
        for line in path.read_text().splitlines():
            fields = line.split()
            # class + >= 3 (x, y) points -> odd count >= 7 (D8).
            assert len(fields) >= 7, f"segment line with {len(fields)} fields in {path}"
            assert len(fields) % 2 == 1, f"even field count in {path}"
            check_common_fields(fields, path)
            lines_checked += 1
    assert lines_checked > 0


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_sampled_images_are_valid_symlinks(label_files, cfg, task):
    # Spot-check the image tree next to the sampled labels: absolute symlinks
    # that resolve to real files (D14).
    task_root = cfg["paths"]["yolo_root"] / f"coco5k-{task}"
    for label_path in sample_label_paths(label_files, task)[:50]:
        split = label_path.parent.name
        image_path = task_root / "images" / split / f"{label_path.stem}.jpg"
        assert image_path.is_symlink(), f"missing symlink {image_path}"
        assert image_path.resolve().is_file(), f"broken symlink {image_path}"
        # D14 requires ABSOLUTE targets — a relative link that happens to
        # resolve would break the moment data/ or datasets/ moved.
        assert Path(os.readlink(image_path)).is_absolute(), f"relative symlink {image_path}"
