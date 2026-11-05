"""
stager.py - Fast, Idempotent, Symlink-Safe Dataset Discovery & Canonical Staging for MMAct

Features:
- Shallow, bounded directory inspection (depth <= 4) over /kaggle/input and local data roots.
- Avoids expensive unbounded recursive file-crawling over remote / mounted filesystems.
- Idempotent: Skips valid existing links and files; repairs broken links; never creates nested links.
- Cycle & loop protection tracking visited directories via device and inode.
- Never deletes the canonical directory as routine setup.
- Produces a comprehensive preflight staging report.
"""

import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from src.data.indexer import MMActIndexer
from src.data.ingestion import DatasetIngestionPipeline


@dataclass
class StagingReport:
    """Preflight diagnostic report of dataset discovery and staging."""

    dataset_root_selected: Path
    canonical_dir: Path
    expected_modalities: List[str]
    discovered_modalities: List[str]
    video_count: int = 0
    sensor_count: int = 0
    split_count: int = 0
    video_subjects: List[str] = field(default_factory=list)
    missing_required_paths: List[str] = field(default_factory=list)
    staging_duration_sec: float = 0.0
    reused: bool = False
    linked_count: int = 0
    copied_count: int = 0
    repaired_count: int = 0

    @property
    def is_valid(self) -> bool:
        return (
            self.video_count > 0
            and self.sensor_count > 0
            and len(self.missing_required_paths) == 0
        )

    def summary(self) -> str:
        lines = [
            "=" * 78,
            " DATASET PREFLIGHT DISCOVERY & STAGING REPORT",
            "=" * 78,
            f"Dataset Root Selected   : {self.dataset_root_selected}",
            f"Canonical Staging Path : {self.canonical_dir}",
            f"Existing Data Reused   : {self.reused}",
            f"Staging Duration       : {self.staging_duration_sec:.3f} seconds",
            f"Links Created/Reused   : {self.linked_count} linked, {self.copied_count} copied, {self.repaired_count} repaired",
            "-" * 78,
            f"Video Subjects Staged  : {sorted(self.video_subjects)}",
            f"Discovered Modalities  : {sorted(self.discovered_modalities)}",
            f"Expected Modalities    : {sorted(self.expected_modalities)}",
            f"Video Files Count      : {self.video_count}",
            f"Sensor Files Count     : {self.sensor_count}",
            f"Split Files Count      : {self.split_count}",
        ]
        if self.missing_required_paths:
            lines.append("-" * 78)
            lines.append("MISSING REQUIRED PATHS:")
            for p in self.missing_required_paths:
                lines.append(f"  - [MISSING] {p}")
        else:
            lines.append("-" * 78)
            lines.append("STATUS: All required MMAct modalities and files verified.")
        lines.append("=" * 78)
        return "\n".join(lines)


class FastMMActStager:
    """Discovers and stages MMAct dataset files using shallow, idempotent inspection."""

    EXPECTED_MODALITIES = [
        "acc_phone_clip",
        "acc_watch_clip",
        "gyro_clip",
    ]

    def __init__(
        self,
        canonical_dir: Path,
        input_roots: Optional[List[Path]] = None,
        max_scan_depth: int = 4,
    ):
        self.canonical_dir = Path(canonical_dir)
        self.input_roots = input_roots or []
        self.max_scan_depth = max_scan_depth

        self.canonical_video = self.canonical_dir / "trimmed" / "video"
        self.canonical_sensors = self.canonical_dir / "trimmed" / "sensors"
        self.canonical_splits = self.canonical_dir / "trimmed" / "splits"

    @staticmethod
    def safe_link_or_copy(src: Path, dst: Path) -> str:
        """
        Idempotently links or copies src to dst.
        Returns: 'reused', 'linked', 'copied', or 'repaired'
        """
        src_resolved = src.resolve()

        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                try:
                    current_target = Path(os.path.realpath(dst)).resolve()
                    if current_target == src_resolved and dst.exists():
                        return "reused"
                    # Broken or mismatched symlink -> remove and repair
                    dst.unlink(missing_ok=True)
                    action = "repaired"
                except OSError:
                    dst.unlink(missing_ok=True)
                    action = "repaired"
            elif dst.is_file() and src.is_file():
                try:
                    if dst.stat().st_size == src.stat().st_size:
                        return "reused"
                    dst.unlink(missing_ok=True)
                    action = "repaired"
                except OSError:
                    action = "copied"
            elif dst.is_dir() and src.is_dir():
                # Directory already populated without being a symlink
                if any(dst.iterdir()):
                    return "reused"
                action = "linked"
            else:
                action = "repaired"
        else:
            action = "linked"

        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.symlink(src_resolved, dst, target_is_directory=src.is_dir())
            return action if action == "repaired" else "linked"
        except (OSError, NotImplementedError):
            if src.is_dir():
                shutil.copytree(src_resolved, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src_resolved, dst)
            return "copied"

    def check_existing_canonical(self) -> Tuple[bool, int, int, int, List[str], List[str]]:
        """
        Quickly checks if canonical_dir is already properly populated.
        Avoids all scanning if already valid.
        """
        if not self.canonical_video.exists() or not self.canonical_sensors.exists():
            return False, 0, 0, 0, [], []

        video_subjects = []
        video_count = 0
        trainval_dir = self.canonical_video / "cross_scene" / "trainval"
        if trainval_dir.exists():
            for s_child in trainval_dir.iterdir():
                if s_child.is_dir():
                    norm_s = MMActIndexer.normalize_identifier(s_child.name, "subject")
                    video_subjects.append(norm_s)
                    # Shallow count videos inside subject cams/scenes
                    for mp4 in s_child.glob("*/*/*/*.mp4"):
                        video_count += 1

        discovered_mods = []
        sensor_count = 0
        if self.canonical_sensors.exists():
            for m_dir in self.canonical_sensors.iterdir():
                if m_dir.is_dir():
                    discovered_mods.append(m_dir.name)
                    for csv in m_dir.glob("*/*/*/*.csv"):
                        sensor_count += 1

        split_count = len(list(self.canonical_splits.glob("*.txt"))) if self.canonical_splits.exists() else 0

        is_valid = (video_count > 0 and sensor_count > 0 and len(video_subjects) > 0)
        return is_valid, video_count, sensor_count, split_count, video_subjects, discovered_mods

    def discover_and_stage(self) -> StagingReport:
        """Executes bounded discovery and idempotent canonical staging."""
        t0 = time.time()

        # 1. Quick check: Is canonical directory already populated and valid?
        is_valid, v_cnt, s_cnt, sp_cnt, v_subjs, d_mods = self.check_existing_canonical()
        if is_valid:
            dur = time.time() - t0
            return StagingReport(
                dataset_root_selected=self.canonical_dir,
                canonical_dir=self.canonical_dir,
                expected_modalities=self.EXPECTED_MODALITIES,
                discovered_modalities=d_mods,
                video_count=v_cnt,
                sensor_count=s_cnt,
                split_count=sp_cnt,
                video_subjects=v_subjs,
                staging_duration_sec=dur,
                reused=True,
            )

        # 2. Gather candidates roots
        roots = [r for r in self.input_roots if r.exists()]
        if not roots:
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                roots.extend([p for p in kaggle_input.iterdir() if p.is_dir()])

        if not roots:
            dur = time.time() - t0
            return StagingReport(
                dataset_root_selected=self.canonical_dir,
                canonical_dir=self.canonical_dir,
                expected_modalities=self.EXPECTED_MODALITIES,
                discovered_modalities=[],
                missing_required_paths=["No input dataset roots found in /kaggle/input or local storage."],
                staging_duration_sec=dur,
                reused=False,
            )

        primary_root = roots[0]
        discovered_videos: Dict[str, Path] = {}
        discovered_sensors: Dict[Tuple[str, str], Path] = {}
        discovered_splits: Dict[str, Path] = {}
        discovered_archives: List[Path] = []

        visited_keys: Set[Union[Tuple[int, int], str]] = set()

        def shallow_scan(curr: Path, depth: int):
            if depth > self.max_scan_depth or not curr.exists() or not curr.is_dir():
                return
            try:
                st = curr.stat()
                key = (st.st_dev, st.st_ino)
            except OSError:
                key = str(curr.resolve())
            if key in visited_keys:
                return
            visited_keys.add(key)

            try:
                children = list(curr.iterdir())
            except (PermissionError, OSError):
                return

            for child in children:
                if child.is_file():
                    suffix = child.suffix.lower()
                    name_lower = child.name.lower()
                    if suffix in {".zip", ".tar", ".gz"}:
                        discovered_archives.append(child)
                    elif suffix == ".txt" and any(k in name_lower for k in ["cross_scene", "cross_view", "train", "val", "test", "split"]):
                        if child.name not in discovered_splits:
                            discovered_splits[child.name] = child
                elif child.is_dir():
                    name_lower = child.name.lower()

                    # Sensor Modality check
                    if name_lower in MMActIndexer.SENSOR_NAME_MAP:
                        raw_map = MMActIndexer.SENSOR_NAME_MAP[name_lower]
                        canon_mod = raw_map if raw_map.endswith("_clip") else f"{raw_map}_clip"
                        try:
                            for s_sub in child.iterdir():
                                if s_sub.is_dir():
                                    s_match = re.search(r"subject_?(\d+)", s_sub.name.lower())
                                    if s_match:
                                        norm_s = f"subject{int(s_match.group(1))}"
                                        if (canon_mod, norm_s) not in discovered_sensors:
                                            discovered_sensors[(canon_mod, norm_s)] = s_sub
                        except (PermissionError, OSError):
                            pass
                        continue

                    # Video Subject check
                    subj_match = re.search(r"subject_?(\d+)", name_lower)
                    is_sensor = any(k in name_lower for k in ["sensor", "acc", "gyro", "ori"])
                    if subj_match and not is_sensor:
                        norm_s = f"subject{int(subj_match.group(1))}"
                        # Check if child contains cams or scenes or nested subject
                        has_video_subdirs = False
                        nested_subj = None
                        try:
                            for sub_c in child.iterdir():
                                if sub_c.is_dir():
                                    sc_lower = sub_c.name.lower()
                                    if any(k in sc_lower for k in ["cam", "view", "scene"]):
                                        has_video_subdirs = True
                                    elif re.match(r"^subject_?\d+$", sc_lower):
                                        nested_subj = sub_c
                        except (PermissionError, OSError):
                            pass

                        if nested_subj is not None and norm_s not in discovered_videos:
                            discovered_videos[norm_s] = nested_subj
                            continue
                        elif has_video_subdirs and norm_s not in discovered_videos:
                            discovered_videos[norm_s] = child
                            continue

                    shallow_scan(child, depth + 1)

        # 3. Perform scan across all input roots
        for r in roots:
            shallow_scan(r, depth=0)

        # 4. If archives detected, unpack via IngestionPipeline
        if discovered_archives:
            dest_downloads = self.canonical_dir.parent / "downloads"
            dest_downloads.mkdir(parents=True, exist_ok=True)
            for arc in discovered_archives:
                dest_arc = dest_downloads / arc.name
                self.safe_link_or_copy(arc, dest_arc)
            ingestion = DatasetIngestionPipeline(
                incoming_dir=dest_downloads,
                raw_storage_dir=self.canonical_dir / "trimmed",
            )
            ingestion.scan_and_ingest()

        # 5. Stage video subjects into canonical structure
        linked_cnt, copied_cnt, repaired_cnt = 0, 0, 0
        canonical_trainval = self.canonical_video / "cross_scene" / "trainval"
        canonical_trainval.mkdir(parents=True, exist_ok=True)

        for norm_s, src_path in discovered_videos.items():
            dst_path = canonical_trainval / norm_s
            stat = self.safe_link_or_copy(src_path, dst_path)
            if stat == "linked":
                linked_cnt += 1
            elif stat == "copied":
                copied_cnt += 1
            elif stat == "repaired":
                repaired_cnt += 1

        # 6. Stage sensor modalities
        self.canonical_sensors.mkdir(parents=True, exist_ok=True)
        for (canon_mod, norm_s), src_path in discovered_sensors.items():
            dst_path = self.canonical_sensors / canon_mod / norm_s
            stat = self.safe_link_or_copy(src_path, dst_path)
            if stat == "linked":
                linked_cnt += 1
            elif stat == "copied":
                copied_cnt += 1
            elif stat == "repaired":
                repaired_cnt += 1

        # 7. Stage splits
        self.canonical_splits.mkdir(parents=True, exist_ok=True)
        for sp_name, src_path in discovered_splits.items():
            dst_path = self.canonical_splits / sp_name
            stat = self.safe_link_or_copy(src_path, dst_path)
            if stat == "linked":
                linked_cnt += 1
            elif stat == "copied":
                copied_cnt += 1
            elif stat == "repaired":
                repaired_cnt += 1

        # 8. Post-staging audit
        _, v_cnt, s_cnt, sp_cnt, v_subjs, d_mods = self.check_existing_canonical()
        missing = []
        if v_cnt == 0:
            missing.append("No video files found in canonical directory.")
        if s_cnt == 0:
            missing.append("No sensor CSV files found in canonical directory.")
        for exp_mod in self.EXPECTED_MODALITIES:
            if exp_mod not in d_mods:
                missing.append(f"Missing required sensor modality: {exp_mod}")

        dur = time.time() - t0
        return StagingReport(
            dataset_root_selected=primary_root,
            canonical_dir=self.canonical_dir,
            expected_modalities=self.EXPECTED_MODALITIES,
            discovered_modalities=d_mods,
            video_count=v_cnt,
            sensor_count=s_cnt,
            split_count=sp_cnt,
            video_subjects=v_subjs,
            missing_required_paths=missing,
            staging_duration_sec=dur,
            reused=False,
            linked_count=linked_cnt,
            copied_count=copied_cnt,
            repaired_count=repaired_cnt,
        )
