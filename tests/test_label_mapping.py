"""
test_label_mapping.py - Tests for verified MMAct label mappings
"""

import json
from pathlib import Path
import pytest

from src.config import (
    VERIFIED_TARGET_CLASSES,
    TARGET_CLASS_TO_IDX,
    IDX_TO_TARGET_CLASS,
    PROJECT_ROOT,
)


def test_verified_classes_count():
    assert len(VERIFIED_TARGET_CLASSES) == 10
    assert len(TARGET_CLASS_TO_IDX) == 10
    assert len(IDX_TO_TARGET_CLASS) == 10


def test_bidirectional_indexing():
    for idx, name in enumerate(VERIFIED_TARGET_CLASSES):
        assert TARGET_CLASS_TO_IDX[name] == idx
        assert IDX_TO_TARGET_CLASS[idx] == name


def test_presence_in_ground_truth_files():
    gt_path = (
        PROJECT_ROOT
        / "challenge-main"
        / "challenge-main"
        / "evaluation"
        / "trimmed_val_scene_gt.json"
    )
    assert gt_path.exists(), f"Ground truth file not found: {gt_path}"

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_labels = set()
    for items in data["annotations"].values():
        for item in items:
            gt_labels.add(item["label"].lower().strip())

    for target_class in VERIFIED_TARGET_CLASSES:
        assert target_class in gt_labels, f"Target class {target_class} missing from ground truth!"
