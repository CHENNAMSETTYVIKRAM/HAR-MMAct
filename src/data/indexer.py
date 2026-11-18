"""
indexer.py - Multimodal and video-only dataset indexing for MMAct.

The default ``index()`` method remains the strict multimodal path: it pairs
videos with all required sensor modalities. ``index_videos()`` is an explicit
video-only path and does not inspect sensor files. Passing an empty required
sensor list to ``index()`` also selects video-only indexing.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import VERIFIED_TARGET_CLASSES, TARGET_CLASS_TO_IDX


@dataclass
class PairedSample:
    """Represents an indexed video sample and optional paired sensors."""

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
    missing_sensors: List[Dict[str, str]] = field(default_factory=list)
    empty_sensor_files: List[str] = field(default_factory=list)
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
            f"Empty Sensor Files        : {len(self.empty_sensor_files)}",
            f"Missing Video Incidents   : {len(self.missing_videos)}",
            f"Unknown Label Files Filtered: {len(self.unknown_label_files)}",
            f"Duplicate IDs Encountered : {len(self.duplicate_ids)}",
            "Class Distribution (Paired):",
        ]
        for cls_name in VERIFIED_TARGET_CLASSES:
            lines.append(f"  - {cls_name:<15}: {self.class_distribution.get(cls_name, 0)} samples")
        lines.append("====================================")
        return "\n".join(lines)


class MMActIndexer:
    """Indexes trimmed MMAct files; multimodal indexing remains strict by default."""

    SENSOR_NAME_MAP = {
        "acc_phone_clip": "acc_phone", "acc_watch_clip": "acc_watch",
        "gyro_clip": "gyro", "orientation_clip": "orientation",
        "acc_phone": "acc_phone", "acc_watch": "acc_watch", "gyro": "gyro",
        "orientation": "orientation", "phone_acc": "acc_phone",
        "phone_acc_clip": "acc_phone", "watch_acc": "acc_watch",
        "watch_acc_clip": "acc_watch", "gyroscope": "gyro",
        "gyroscope_clip": "gyro", "ori": "orientation", "ori_clip": "orientation",
    }

    def __init__(self, dataset_root: Path, required_sensor_modalities: Optional[List[str]] = None,
                 target_classes: Optional[List[str]] = None):
        self.dataset_root = Path(dataset_root)
        self.required_sensors = (
            ["acc_phone", "acc_watch", "gyro"]
            if required_sensor_modalities is None
            else list(required_sensor_modalities)
        )
        self.target_classes = set(target_classes or VERIFIED_TARGET_CLASSES)

    @staticmethod
    def normalize_action_name(stem: str) -> str:
        name = stem.strip().lower().replace(" ", "_")
        return re.sub(r"[\(_\s]*\d+[\)]*$", "", name).strip("_")

    @staticmethod
    def normalize_identifier(identifier: str, prefix: str) -> str:
        match = re.search(r"\d+", identifier)
        return f"{prefix}{int(match.group(0))}" if match else identifier.lower()

    @staticmethod
    def parse_video_path(path: Path) -> Optional[Tuple[str, str, str, str, str]]:
        parts = [p.lower() for p in path.parts]
        subj_raw = next((p for p in parts if re.match(r"^(subject|sub)_?\d+$", p)), None)
        cam_raw = next((p for p in parts if re.match(r"^(cam|camera|view)_?\d+$|^ego$", p)), None)
        scene_raw = next((p for p in parts if re.match(r"^scene_?\d+$", p)), None)
        session_raw = next((p for p in parts if re.match(r"^session_?\d+$", p)), None)
        action = MMActIndexer.normalize_action_name(path.stem)
        if subj_raw and cam_raw and scene_raw and session_raw and action:
            subject = MMActIndexer.normalize_identifier(subj_raw, "subject")
            camera = "ego" if cam_raw == "ego" else MMActIndexer.normalize_identifier(cam_raw, "cam")
            scene = MMActIndexer.normalize_identifier(scene_raw, "scene")
            session = MMActIndexer.normalize_identifier(session_raw, "session")
            return subject, camera, scene, session, action
        return None

    def parse_sensor_path(self, path: Path) -> Optional[Tuple[str, str, str, str, str]]:
        parts = [p.lower() for p in path.parts]
        sensor_type = next((self.SENSOR_NAME_MAP[p] for p in parts if p in self.SENSOR_NAME_MAP), None)
        subj_raw = next((p for p in parts if re.match(r"^(subject|sub)_?\d+$", p)), None)
        scene_raw = next((p for p in parts if re.match(r"^scene_?\d+$", p)), None)
        session_raw = next((p for p in parts if re.match(r"^session_?\d+$", p)), None)
        action = self.normalize_action_name(path.stem)
        if sensor_type and subj_raw and scene_raw and session_raw and action:
            return (sensor_type, self.normalize_identifier(subj_raw, "subject"),
                    self.normalize_identifier(scene_raw, "scene"),
                    self.normalize_identifier(session_raw, "session"), action)
        return None

    @staticmethod
    def find_files_recursively(root: Path, extensions: Set[str]) -> List[Path]:
        """Find matching files while safely traversing symlinks with cycle protection."""
        results: List[Path] = []
        root_path = Path(root)
        if not root_path.exists():
            return results
        visited_dirs: Set[Any] = set()
        for dirpath, dirnames, filenames in os.walk(str(root_path), followlinks=True):
            try:
                st = os.stat(dirpath)
                dir_key = (st.st_dev, st.st_ino)
            except OSError:
                dir_key = os.path.realpath(dirpath)
            if dir_key in visited_dirs:
                dirnames.clear()
                continue
            visited_dirs.add(dir_key)
            results.extend(Path(dirpath) / fname for fname in filenames
                           if os.path.splitext(fname)[1].lower() in extensions)
        return results

    def index_videos(self) -> IndexerReport:
        """Index valid target-class MP4 clips without discovering or requiring sensors."""
        report = IndexerReport()
        if not self.dataset_root.exists():
            return report
        seen_ids: Set[str] = set()
        for video_path in self.find_files_recursively(self.dataset_root, {".mp4"}):
            report.total_video_files_found += 1
            parsed = self.parse_video_path(video_path)
            if not parsed:
                continue
            subject, camera, scene, session, action = parsed
            if action not in self.target_classes:
                report.unknown_label_files.append(str(video_path))
                continue
            sample_id = f"{subject}_{scene}_{session}_{camera}_{action}"
            if sample_id in seen_ids:
                report.duplicate_ids.append(sample_id)
                continue
            seen_ids.add(sample_id)
            report.paired_samples.append(PairedSample(
                sample_id=sample_id, action=action, label_idx=TARGET_CLASS_TO_IDX[action],
                subject=subject, scene=scene, session=session, cam=camera,
                video_path=video_path, sensor_paths={},
            ))
            report.class_distribution[action] = report.class_distribution.get(action, 0) + 1
        report.paired_samples_count = len(report.paired_samples)
        return report

    def index(self) -> IndexerReport:
        """Index videos, pairing all configured sensors unless in explicit video-only mode."""
        if not self.required_sensors:
            # This path deliberately avoids even searching for sensor files.
            return self.index_videos()

        report = IndexerReport()
        if not self.dataset_root.exists():
            return report

        video_map: Dict[Tuple[str, str, str, str, str], Path] = {}
        for vid_file in self.find_files_recursively(self.dataset_root, {".mp4"}):
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

        sensor_map: Dict[Tuple[str, str, str, str], Dict[str, Path]] = {}
        video_subjects = {k[0] for k in video_map.keys()}
        sensor_files: List[Path] = []
        possible_sensor_roots = [self.dataset_root / "trimmed" / "sensors", self.dataset_root / "trimmed" / "sensor",
                                 self.dataset_root / "sensors", self.dataset_root / "sensor", self.dataset_root / "trimmed_sensor"]
        scanned_targeted = False
        for s_root in possible_sensor_roots:
            if s_root.exists() and s_root != self.dataset_root:
                scanned_targeted = True
                for sub_child in [s_root / "sensor", s_root / "sensors"]:
                    if sub_child.is_dir():
                        s_root = sub_child
                        break
                for clip_dir in s_root.iterdir():
                    if clip_dir.is_dir():
                        for v_subj in video_subjects:
                            target_s_dir = clip_dir / v_subj
                            if target_s_dir.exists():
                                sensor_files.extend(self.find_files_recursively(target_s_dir, {".csv"}))
                break
        if not scanned_targeted or len(sensor_files) == 0:
            sensor_files = self.find_files_recursively(self.dataset_root, {".csv"})

        for sen_file in sensor_files:
            report.total_sensor_files_found += 1
            if sen_file.stat().st_size == 0:
                report.empty_sensor_files.append(str(sen_file))
                continue
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
                report.duplicate_ids.append(f"sensor:{session_key}:{sensor_type}")
            sensor_map[session_key][sensor_type] = sen_file

        seen_sample_ids: Set[str] = set()
        for (subj, cam, scene, session, action), vid_path in video_map.items():
            session_key = (subj, scene, session, action)
            matched_sensors = sensor_map.get(session_key, {})
            missing_req = [s for s in self.required_sensors if s not in matched_sensors]
            if missing_req:
                report.missing_sensors.append({"video_path": str(vid_path), "subject": subj, "scene": scene,
                    "session": session, "cam": cam, "action": action, "missing_modalities": missing_req})
                continue
            sample_id = f"{subj}_{scene}_{session}_{cam}_{action}"
            if sample_id in seen_sample_ids:
                report.duplicate_ids.append(sample_id)
                continue
            seen_sample_ids.add(sample_id)
            report.paired_samples.append(PairedSample(sample_id, action, TARGET_CLASS_TO_IDX[action], subj,
                scene, session, cam, vid_path, {s: matched_sensors[s] for s in self.required_sensors}))
            report.class_distribution[action] = report.class_distribution.get(action, 0) + 1

        paired_session_keys = {(p.subject, p.scene, p.session, p.action) for p in report.paired_samples}
        for (subj, scene, session, action), sens_dict in sensor_map.items():
            if (subj, scene, session, action) not in paired_session_keys:
                report.missing_videos.append({"subject": subj, "scene": scene, "session": session,
                    "action": action, "available_sensors": list(sens_dict.keys())})
        report.paired_samples_count = len(report.paired_samples)
        return report
