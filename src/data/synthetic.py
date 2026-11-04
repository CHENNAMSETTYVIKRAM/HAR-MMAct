"""
synthetic.py - Synthetic Smoke-Test Fixture Generator for HAR_MMAct

Generates a minimal, strictly labeled synthetic dataset following MMAct naming conventions.
CRITICAL: ALL ARTIFACTS ARE MARKED WITH [SYNTHETIC] METADATA TO PREVENT FABRICATION OR CONFUSION.
Used solely for smoke-testing indexing, preprocessing, model forward passes, and training loops.
"""

import json
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import pandas as pd

from src.config import VERIFIED_TARGET_CLASSES


def generate_synthetic_fixture(
    output_root: Path,
    classes: Optional[List[str]] = None,
    num_subjects: int = 3,
    num_cams: int = 2,
    include_test_anomalies: bool = True,
) -> Path:
    """
    Creates a synthetic mini MMAct dataset for smoke testing.
    Directory structure:
      <output_root>/
        ├── SYNTHETIC_METADATA.json
        ├── trimmed/
        │   ├── video/
        │   │   └── cross_scene/trainval/subjectX/camY/sceneZ/session1/<action>.mp4
        │   └── sensors/
        │       ├── acc_phone_clip/subjectX/sceneZ/session1/<action>.csv
        │       ├── acc_watch_clip/subjectX/sceneZ/session1/<action>.csv
        │       └── gyro_clip/subjectX/sceneZ/session1/<action>.csv
    """
    target_classes = classes or VERIFIED_TARGET_CLASSES
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    metadata_path = root / "SYNTHETIC_METADATA.json"
    metadata = {
        "is_synthetic": True,
        "warning": "DO NOT USE FOR BENCHMARKING. THIS IS A SYNTHETIC SMOKE-TEST FIXTURE ONLY.",
        "target_classes": target_classes,
        "num_subjects": num_subjects,
        "num_cams": num_cams,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Base trimmed directories
    video_base = root / "trimmed" / "video" / "cross_scene" / "trainval"
    sensor_base = root / "trimmed" / "sensors"

    scenes = ["scene1", "scene2"]
    session = "session1"

    for s_idx in range(1, num_subjects + 1):
        subj = f"subject{s_idx}"
        for sc_idx, scene in enumerate(scenes, 1):
            for action in target_classes:
                # 1. Generate paired sensor CSVs (view-independent)
                # Timestamp start
                base_time = pd.Timestamp(
                    "2026-11-04 12:00:00"
                ) + pd.Timedelta(seconds=(s_idx * 100 + sc_idx * 10))

                # Phone accelerometer (100 Hz, 120 samples = 1.2 sec)
                t_phone = [
                    (base_time + pd.Timedelta(milliseconds=10 * i)).strftime(
                        "%Y%m%d_%H:%M:%S.%f"
                    )[:-3]
                    for i in range(120)
                ]
                acc_p_data = {
                    "time": t_phone,
                    "x": np.sin(np.linspace(0, 3.14 * 2, 120)) * 2.0
                    + np.random.normal(0, 0.1, 120),
                    "y": np.cos(np.linspace(0, 3.14 * 2, 120)) * 2.0
                    + 9.8
                    + np.random.normal(0, 0.1, 120),
                    "z": np.random.normal(0, 0.5, 120),
                }
                phone_dir = sensor_base / "acc_phone_clip" / subj / scene / session
                phone_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(acc_p_data).to_csv(
                    phone_dir / f"{action}.csv", index=False
                )

                # Watch accelerometer (100 Hz, 120 samples)
                acc_w_data = {
                    "time": t_phone,
                    "x": np.cos(np.linspace(0, 3.14 * 4, 120)) * 3.0
                    + np.random.normal(0, 0.1, 120),
                    "y": np.sin(np.linspace(0, 3.14 * 4, 120)) * 3.0
                    + 9.8
                    + np.random.normal(0, 0.1, 120),
                    "z": np.random.normal(0, 0.4, 120),
                }
                watch_dir = sensor_base / "acc_watch_clip" / subj / scene / session
                watch_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(acc_w_data).to_csv(
                    watch_dir / f"{action}.csv", index=False
                )

                # Gyroscope (50 Hz, 60 samples = 1.2 sec)
                t_gyro = [
                    (base_time + pd.Timedelta(milliseconds=20 * i)).strftime(
                        "%Y%m%d_%H:%M:%S.%f"
                    )[:-3]
                    for i in range(60)
                ]
                gyro_data = {
                    "time": t_gyro,
                    "x": np.sin(np.linspace(0, 3.14, 60)) * 10.0
                    + np.random.normal(0, 0.2, 60),
                    "y": np.cos(np.linspace(0, 3.14, 60)) * 10.0
                    + np.random.normal(0, 0.2, 60),
                    "z": np.random.normal(0, 1.0, 60),
                }
                gyro_dir = sensor_base / "gyro_clip" / subj / scene / session
                gyro_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(gyro_data).to_csv(
                    gyro_dir / f"{action}.csv", index=False
                )

                # 2. Generate paired videos for each camera view
                for c_idx in range(1, num_cams + 1):
                    cam = f"cam{c_idx}"
                    vid_dir = video_base / subj / cam / scene / session
                    vid_dir.mkdir(parents=True, exist_ok=True)
                    vid_path = vid_dir / f"{action}.mp4"

                    # Write small synthetic 16-frame 64x64 video
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        str(vid_path), fourcc, 15.0, (64, 64)
                    )
                    # Color pattern based on class index
                    cls_val = target_classes.index(action) * 25
                    for f in range(16):
                        frame = np.zeros((64, 64, 3), dtype=np.uint8)
                        frame[:, :, 0] = (cls_val + f * 5) % 256
                        frame[:, :, 1] = (128 + f * 3) % 256
                        frame[:, :, 2] = (s_idx * 50) % 256
                        out.write(frame)
                    out.release()

    # 3. Add intentional anomalies for testing diagnostic filtering
    if include_test_anomalies:
        # A. Unknown label: 'drinking.mp4' (not in the 10 target classes)
        anomaly_vid = (
            video_base
            / "subject1"
            / "cam1"
            / "scene1"
            / session
            / "drinking.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(anomaly_vid), fourcc, 15.0, (64, 64))
        for _ in range(8):
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out.release()

        # B. Missing sensor: video exists for 'subject1', 'cam1', 'scene2', 'closing.mp4', but no sensors exist
        anomaly_vid2 = (
            video_base
            / "subject1"
            / "cam1"
            / "scene2"
            / session
            / "crouching_no_sensor.mp4"
        )
        out2 = cv2.VideoWriter(str(anomaly_vid2), fourcc, 15.0, (64, 64))
        for _ in range(8):
            out2.write(np.zeros((64, 64, 3), dtype=np.uint8))
        out2.release()

        # C. Missing video: sensor exists for 'jumping_no_video', but no video exists
        orphan_sensor_dir = (
            sensor_base / "acc_phone_clip" / "subject1" / "scene1" / session
        )
        pd.DataFrame(
            {"time": ["20261104_12:00:00.000"], "x": [0.0], "y": [9.8], "z": [0.0]}
        ).to_csv(orphan_sensor_dir / "jumping_no_video.csv", index=False)

    return root
