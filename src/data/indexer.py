"""
indexer.py - Multimodal Dataset Indexer for MMAct

Scans genuine (or synthetic) MMAct directory structures, extracts sample metadata,
pairs video clips with corresponding wearable sensor recordings based on source
identifiers (subject, scene, session, action, cam), and outputs strict diagnostic
reports (missing video, missing sensors, unknown labels, duplicate IDs).
Filters exclusively to verified target classes.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.config import (
    VERIFIED_TARGET_CLASSES,
    TARGET_CLASS_TO_IDX,
)


@dataclass
class PairedSample:
    """Represents a strictly matched video-sensor multimodal sample."""

    sample_id: str
    action: str
    label_idx: int
    subject: str
    scene: str
    session: str
    cam: str
    video_path: Path
    sensor_paths: Dict[str, Path]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "action": self.action,
            "label_idx": self.label_idx,
            "subject": self.subject,
            "scene": self.scene,
            "session": self.session,
            "cam": self.cam,
            "video_path": str(self.video_path),
            "sensor_paths": {k: str(v) for k, v in self.sensor_paths.items()},
        }


@dataclass
class IndexerReport:
    """Detailed diagnostic report of dataset indexing."""

    total_video_files_found: int = 0
    total_sensor_files_found: int = 0
    paired_samples_count: int = 0
    paired_samples: List[PairedSample] = field(default_factory=list)

    # Diagnostic anomaly lists
    missing_sensors: List[Dict[str, str]] = field(default_factory=list)
    missing_videos: List[Dict[str, str]] = field(default_factory=list)
    unknown_label_files: List[str] = field(default_factory=list)
    duplicate_ids: List[str] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== MMAct Dataset Indexer Report ===",
            f"Total Video Files Found   : {self.total_video_files_found}",
            f"Total Sensor Files Found  : {self.total_sensor_files_found}",
            f"Paired Multimodal Samples : {self.paired_samples_count}",
            f"Missing Sensor Incidents  : {len(self.missing_sensors)}",
            f"Missing Video Incidents   : {len(self.missing_videos)}",
            f"Unknown Label Files Filtered: {len(self.unknown_label_files)}",
            f"Duplicate IDs Encountered : {len(self.duplicate_ids)}",
            "Class Distribution (Paired):",
        ]
        for cls_name in VERIFIED_TARGET_CLASSES:
            lines.append(
                f"  - {cls_name:<15}: {self.class_distribution.get(cls_name, 0)}"
            )
        lines.append("====================================")
        return "\n".join(lines)


class MMActIndexer:
    """Indexes trimmed MMAct video and sensor files and pairs them strictly."""

    # Robust sensor naming map supporting official names and archive variations
    SENSOR_NAME_MAP = {
        "acc_phone_clip": "acc_phone",
        "acc_watch_clip": "acc_watch",
        "gyro_clip": "gyro",
        "orientation_clip": "orientation",
        "acc_phone": "acc_phone",
        "acc_watch": "acc_watch",
        "gyro": "gyro",
        "orientation": "orientation",
        "phone_acc": "acc_phone",
        "phone_acc_clip": "acc_phone",
        "watch_acc": "acc_watch",
        "watch_acc_clip": "acc_watch",
        "gyroscope": "gyro",
        "gyroscope_clip": "gyro",
        "ori": "orientation",
        "ori_clip": "orientation",
    }

    def __init__(
        self,
        dataset_root: Path,
        required_sensor_modalities: Optional[List[str]] = None,
        target_classes: Optional[List[str]] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.required_sensors = required_sensor_modalities or [
            "acc_phone",
            "acc_watch",
            "gyro",
        ]
        self.target_classes = set(target_classes or VERIFIED_TARGET_CLASSES)

    @staticmethod
    def normalize_action_name(stem: str) -> str:
        """
        Normalizes action filenames, resolving casing, spaces, and punctuation:
        e.g. 'Carrying_light' -> 'carrying_light', 'waving hand' -> 'waving_hand'
        """
        name = stem.strip().lower().replace(" ", "_")
        # Strip trailing numbers like 'walking_1' or 'walking(1)'
        name = re.sub(r"[\(_\s]*\d+[\)]*$", "", name).strip("_")
        return name

    @staticmethod
    def normalize_identifier(identifier: str, prefix: str) -> str:
        """
        Normalizes e.g. 'subject01' -> 'subject1', 'cam_1' or 'view1' -> 'cam1'
        """
        match = re.search(r"\d+", identifier)
        if match:
            num = int(match.group(0))
            return f"{prefix}{num}"
        return identifier.lower()

    def parse_video_path(
        self, path: Path
    ) -> Optional[Tuple[str, str, str, str, str]]:
        """
        Parses video path to extract (subject, cam, scene, session, action).
        Matches pattern: .../subjectX/.../camY/.../sceneZ/.../sessionW/<action>.mp4
        Supports view1..view4, camera1..camera4, and leading-zero numbers.
        """
        parts = [p.lower() for p in path.parts]

        subj_raw = next(
            (p for p in parts if re.match(r"^(subject|sub)_?\d+$", p)),
            None,
        )
        cam_raw = next(
            (p for p in parts if re.match(r"^(cam|camera|view)_?\d+$|^ego$", p)),
            None,
        )
        scene_raw = next(
            (p for p in parts if re.match(r"^scene_?\d+$", p)),
            None,
        )
        session_raw = next(
            (p for p in parts if re.match(r"^session_?\d+$", p)),
            None,
        )
        action = self.normalize_action_name(path.stem)

        if subj_raw and cam_raw and scene_raw and session_raw and action:
            subj = self.normalize_identifier(subj_raw, "subject")
            cam = "ego" if cam_raw == "ego" else self.normalize_identifier(cam_raw, "cam")
            scene = self.normalize_identifier(scene_raw, "scene")
            session = self.normalize_identifier(session_raw, "session")
            return (subj, cam, scene, session, action)
        return None

    def parse_sensor_path(
        self, path: Path
    ) -> Optional[Tuple[str, str, str, str, str]]:
        """
        Parses sensor path to extract (sensor_type, subject, scene, session, action).
        Matches pattern: .../<sensor_clip>/subjectX/sceneZ/sessionW/<action>.csv
        """
        parts = [p.lower() for p in path.parts]

        sensor_type = None
        for p in parts:
            if p in self.SENSOR_NAME_MAP:
                sensor_type = self.SENSOR_NAME_MAP[p]
                break

        subj_raw = next(
            (p for p in parts if re.match(r"^(subject|sub)_?\d+$", p)),
            None,
        )
        scene_raw = next(
            (p for p in parts if re.match(r"^scene_?\d+$", p)),
            None,
        )
        session_raw = next(
            (p for p in parts if re.match(r"^session_?\d+$", p)),
            None,
        )
        action = self.normalize_action_name(path.stem)

        if sensor_type and subj_raw and scene_raw and session_raw and action:
            subj = self.normalize_identifier(subj_raw, "subject")
            scene = self.normalize_identifier(scene_raw, "scene")
            session = self.normalize_identifier(session_raw, "session")
            return (sensor_type, subj, scene, session, action)
        return None

    def index(self) -> IndexerReport:
        """Scan directory and produce paired dataset report."""
        report = IndexerReport()

        if not self.dataset_root.exists():
            return report

        # 1. Discover all videos
        video_map: Dict[Tuple[str, str, str, str, str], Path] = {}
        for vid_file in self.dataset_root.glob("**/*.mp4"):
            report.total_video_files_found += 1
            parsed = self.parse_video_path(vid_file)
            if not parsed:
                continue
            subj, cam, scene, session, action = parsed

            if action not in self.target_classes:
                report.unknown_label_files.append(str(vid_file))
                continue

            key = (subj, cam, scene, session, action)
            if key in video_map:
                report.duplicate_ids.append(f"video:{key}")
            video_map[key] = vid_file

        # 2. Discover all sensors
        # Key: (subject, scene, session, action) -> {sensor_type: path}
        sensor_map: Dict[Tuple[str, str, str, str], Dict[str, Path]] = {}
        for sen_file in self.dataset_root.glob("**/*.csv"):
            report.total_sensor_files_found += 1
            parsed = self.parse_sensor_path(sen_file)
            if not parsed:
                continue
            sensor_type, subj, scene, session, action = parsed

            if action not in self.target_classes:
                report.unknown_label_files.append(str(sen_file))
                continue

            session_key = (subj, scene, session, action)
            if session_key not in sensor_map:
                sensor_map[session_key] = {}

            if sensor_type in sensor_map[session_key]:
                report.duplicate_ids.append(
                    f"sensor:{session_key}:{sensor_type}"
                )
            sensor_map[session_key][sensor_type] = sen_file

        # 3. Pair video and sensors
        seen_sample_ids: Set[str] = set()

        for (
            subj,
            cam,
            scene,
            session,
            action,
        ), vid_path in video_map.items():
            session_key = (subj, scene, session, action)
            matched_sensors = sensor_map.get(session_key, {})

            # Check if all required sensor modalities exist
            missing_req = [
                s for s in self.required_sensors if s not in matched_sensors
            ]
            if missing_req:
                report.missing_sensors.append(
                    {
                        "video_path": str(vid_path),
                        "subject": subj,
                        "scene": scene,
                        "session": session,
                        "cam": cam,
                        "action": action,
                        "missing_modalities": missing_req,
                    }
                )
                continue

            sample_id = f"{subj}_{scene}_{session}_{cam}_{action}"
            if sample_id in seen_sample_ids:
                report.duplicate_ids.append(sample_id)
                continue
            seen_sample_ids.add(sample_id)

            paired = PairedSample(
                sample_id=sample_id,
                action=action,
                label_idx=TARGET_CLASS_TO_IDX[action],
                subject=subj,
                scene=scene,
                session=session,
                cam=cam,
                video_path=vid_path,
                sensor_paths={
                    s: matched_sensors[s] for s in self.required_sensors
                },
            )
            report.paired_samples.append(paired)
            report.class_distribution[action] = (
                report.class_distribution.get(action, 0) + 1
            )

        # 4. Check for sensor sessions that had no video
        paired_session_keys = {
            (p.subject, p.scene, p.session, p.action)
            for p in report.paired_samples
        }
        for (
            subj,
            scene,
            session,
            action,
        ), sens_dict in sensor_map.items():
            if (subj, scene, session, action) not in paired_session_keys:
                report.missing_videos.append(
                    {
                        "subject": subj,
                        "scene": scene,
                        "session": session,
                        "action": action,
                        "available_sensors": list(sens_dict.keys()),
                    }
                )

        report.paired_samples_count = len(report.paired_samples)
        return report
