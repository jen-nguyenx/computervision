"""Shared pytest fixtures for the COCO5K pipeline tests.

The instances json is ~450MB and takes ~7 seconds to parse, so it is loaded
exactly once per pytest session (session-scoped fixture) and shared by every
test module that needs it.
"""

import sys
from pathlib import Path

import pytest

# Make `import src...` work no matter where pytest is invoked from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.common import load_config, load_instances  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    """The pipeline policy config (configs/pipeline_coco.yaml), paths absolute."""
    return load_config()


@pytest.fixture(scope="session")
def instances(cfg):
    """The raw COCO instances json, loaded once for the whole session (~7s)."""
    return load_instances(cfg)


@pytest.fixture(scope="session")
def label_files(cfg):
    """{task: {split: sorted list of label .txt paths}} for both YOLO trees (D14)."""
    tree = {}
    for task in ("detect", "segment"):
        task_root = cfg["paths"]["yolo_root"] / f"coco5k-{task}"
        tree[task] = {
            split: sorted((task_root / "labels" / split).glob("*.txt"))
            for split in ("train", "val", "test")
        }
    return tree
