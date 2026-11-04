"""
audit_dataset.py - Executable Dataset & Label Audit for MMAct HAR Project

This script audits available dataset files, annotations, evaluation splits,
and verifies target activity classes against official ground truth files.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

# Requested target classes
TARGET_CLASSES = [
    "Walking",
    "Jumping",
    "Running",
    "Standing",
    "Waving hand",
    "Kicking",
    "Sitting",
    "Crouching",
    "Using phone",
    "Pointing",
]

# Standard ground truth files in the challenge repository
EVAL_FILES = {
    "trimmed_val_scene": "challenge-main/challenge-main/evaluation/trimmed_val_scene_gt.json",
    "trimmed_val_view": "challenge-main/challenge-main/evaluation/trimmed_val_view_gt.json",
    "untrimmed_val": "challenge-main/challenge-main/evaluation/untrimmed_val_gt.json",
}


def audit_workspace(workspace_root: Path):
    print("=" * 70)
    print("MMAct Dataset & Label Audit")
    print(f"Workspace Root: {workspace_root}")
    print("=" * 70)

    # 1. Check workspace inventory
    print("\n[1] Checking local files & dataset assets...")
    found_assets = {
        "videos": list(workspace_root.glob("**/*.mp4")),
        "sensors": list(workspace_root.glob("**/*.csv")),
        "archives": [
            p
            for p in workspace_root.glob("**/*")
            if p.suffix.lower() in [".zip", ".tar", ".gz", ".7z", ".rar"]
        ],
        "annotations_json": list(
            (workspace_root / "challenge-main/challenge-main/evaluation").glob(
                "*.json"
            )
        ),
        "source_files": list(workspace_root.glob("sources.txt")),
    }

    print(f"  - Video files (.mp4) found: {len(found_assets['videos'])}")
    for v in found_assets["videos"]:
        rel = v.relative_to(workspace_root)
        size_kb = v.stat().st_size / 1024
        print(f"      * {rel} ({size_kb:.1f} KB)")

    print(f"  - Sensor logs (.csv) found: {len(found_assets['sensors'])}")
    print(f"  - Dataset archive files found: {len(found_assets['archives'])}")
    print(
        f"  - Evaluation/Ground-truth JSON files found: {len(found_assets['annotations_json'])}"
    )
    for aj in found_assets["annotations_json"]:
        rel = aj.relative_to(workspace_root)
        size_kb = aj.stat().st_size / 1024
        print(f"      * {rel} ({size_kb:.1f} KB)")

    # 2. Inspect ground truth annotations
    print("\n[2] Auditing Ground Truth Annotation Files...")
    gt_data = {}
    gt_labels = {}
    gt_counts = {}

    for name, rel_path in EVAL_FILES.items():
        full_path = workspace_root / rel_path
        if not full_path.exists():
            print(f"  [MISSING] {rel_path}")
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", {})
        counts = Counter()
        for video_id, items in annotations.items():
            for item in items:
                label = item.get("label", "").strip()
                counts[label] += 1

        gt_data[name] = annotations
        gt_labels[name] = set(counts.keys())
        gt_counts[name] = counts

        print(
            f"  - {name}: {len(annotations)} samples, {len(counts)} distinct labels"
        )

    # 3. Target Class Mapping & Verification
    print("\n[3] Target Class Verification:")
    print(
        f"{'Requested Label':<16} {'Normalized':<16} {'Trimmed Scene':<15} {'Trimmed View':<15} {'Untrimmed':<12} {'Status'}"
    )
    print("-" * 88)

    mapping_results = []
    all_verified = True

    for target in TARGET_CLASSES:
        norm = target.lower().replace(" ", "_")
        c_scene = norm if norm in gt_labels.get("trimmed_val_scene", set()) else None
        c_view = norm if norm in gt_labels.get("trimmed_val_view", set()) else None
        c_untrimmed = norm if norm in gt_labels.get("untrimmed_val", set()) else None

        scene_str = (
            f"YES ({gt_counts['trimmed_val_scene'][norm]})" if c_scene else "NO"
        )
        view_str = f"YES ({gt_counts['trimmed_val_view'][norm]})" if c_view else "NO"
        untrimmed_str = (
            f"YES ({gt_counts['untrimmed_val'][norm]})" if c_untrimmed else "NO"
        )

        if c_scene and c_view and c_untrimmed:
            status = "VERIFIED"
        else:
            status = "INCOMPLETE"
            all_verified = False

        print(
            f"{target:<16} {norm:<16} {scene_str:<15} {view_str:<15} {untrimmed_str:<12} {status}"
        )
        mapping_results.append(
            {
                "requested": target,
                "normalized": norm,
                "scene_count": gt_counts.get("trimmed_val_scene", {}).get(
                    norm, 0
                ),
                "view_count": gt_counts.get("trimmed_val_view", {}).get(
                    norm, 0
                ),
                "untrimmed_count": gt_counts.get("untrimmed_val", {}).get(
                    norm, 0
                ),
                "status": status,
            }
        )

    # 4. Check for related / ambiguous classes
    print("\n[4] Class Disambiguation & Related Labels in MMAct:")
    all_known_classes = sorted(
        list(set.union(*gt_labels.values())) if gt_labels else []
    )
    potential_confusions = {
        "sitting": [c for c in all_known_classes if "sit" in c or "stand" in c],
        "standing": [
            c for c in all_known_classes if "stand" in c or "sit" in c
        ],
        "using_phone": [c for c in all_known_classes if "phone" in c],
        "walking": [
            c
            for c in all_known_classes
            if "walk" in c or "run" in c or "loiter" in c
        ],
    }
    for req_key, related in potential_confusions.items():
        print(f"  - Target '{req_key}': Related MMAct labels found -> {related}")

    # 5. Summary & Blockers
    print("\n[5] Audit Summary & Blockers:")
    if all_verified:
        print(
            "  [PASSED] All 10 requested target classes exist in official MMAct ground truth files."
        )
    else:
        print(
            "  [FAILED] One or more target classes could not be verified in ground truth files."
        )

    if len(found_assets["videos"]) <= 1 and len(found_assets["sensors"]) == 0:
        print(
            "  [ALERT] Full dataset video and sensor archives are NOT currently downloaded in workspace."
        )
        print(
            "          Only 1 sample clip ('carrying.mp4') is present locally."
        )
        print(
            "          Minimal subset archives or synthetic smoke-test paired data will be needed for Phase 2."
        )

    print("=" * 70)
    return mapping_results


if __name__ == "__main__":
    workspace = Path(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )
    audit_workspace(workspace)
