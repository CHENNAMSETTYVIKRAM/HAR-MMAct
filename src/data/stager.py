"""
stager.py - Fast, Idempotent, Symlink-Safe Dataset Discovery & Canonical Staging for MMAct

Features:
- Dynamically discovers multi-dataset mounts under /kaggle/input (e.g. mmact-trimmed, trimmed-scene-video).
- Handles nested directory structures (/kaggle/input/datasets/<user>/<slug> and /kaggle/input/<slug>).
- Validates the nested trimmed_sensor/sensor/ directory across all 4 sensor modalities:
  (acc_phone_clip, acc_watch_clip, gyro_clip, orientation_clip).
- Locates official MMAct split files (trimmed_splits/splits).
- Merged recording inventory: indexes every MP4 across all discovered sources by recording identity.
- Full-file SHA-256 duplicate verification across every source containing the same recording key.
- Deduplicates identical files; records conflicts and blocks training if any sources differ.
- Preserves all unique recordings from primary and secondary sources.
- Safe canonical reuse: verifies full-file SHA-256 for files and expected contents for directories.
- Idempotent: Never overwrites or modifies read-only input roots; checks symlink support; repairs broken links.
- Emits machine-readable manifests and a concise preflight summary report.
"""

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.data.indexer import MMActIndexer
from src.data.ingestion import DatasetIngestionPipeline

EXPECTED_MODALITIES = [
    "acc_phone_clip",
    "acc_watch_clip",
    "gyro_clip",
    "orientation_clip",
]

EXPECTED_OFFICIAL_SPLITS = [
    "trimmed_x-scene_train.txt",
    "trimmed_x-scene_val.txt",
    "trimmed_x-view_train.txt",
    "trimmed_x-view_val.txt",
]


def full_file_hash(file_path: Path) -> str:
    """
    Computes full SHA-256 of an entire file for definitive byte-level identity verification.
    Reads in 1MB chunks to handle large video files efficiently.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def fast_file_hash(file_path: Path) -> str:
    """
    Computes sampled SHA-256 for preliminary filtering.
    NOTE: Never use as definitive proof of identity; always verify with full_file_hash().
    """
    hasher = hashlib.sha256()
    size = file_path.stat().st_size
    hasher.update(f"size:{size}:".encode("utf-8"))
    with open(file_path, "rb") as f:
        if size <= 2 * 1024 * 1024:
            hasher.update(f.read())
        else:
            # Hash first 512KB and last 512KB
            hasher.update(f.read(512 * 1024))
            f.seek(max(0, size - 512 * 1024))
            hasher.update(f.read(512 * 1024))
    return hasher.hexdigest()


def discover_kaggle_input_roots(
    candidate_roots: Optional[Union[List[Union[str, Path]], str, Path]] = None,
    base_dir: Union[str, Path] = Path("/kaggle/input"),
) -> List[Path]:
    """
    Discovers all attached dataset roots under /kaggle/input or passed candidate roots.
    Handles direct mounts, intermediate /kaggle/input/datasets/<user>/<slug> paths, and local paths.
    """
    discovered: List[Path] = []
    seen: Set[str] = set()

    def add_root(p: Path):
        try:
            res = p.resolve()
            key = str(res)
        except OSError:
            key = str(p)
        if key not in seen and p.exists() and p.is_dir():
            seen.add(key)
            discovered.append(p)

    target_base = Path(base_dir)
    items_to_check: List[Path] = []

    if candidate_roots is not None:
        if isinstance(candidate_roots, (str, Path)):
            p = Path(candidate_roots)
            if p.is_dir() and (p.name.lower() == "input" or (p / "datasets").exists()):
                target_base = p
            else:
                items_to_check.append(p)
        else:
            items_to_check.extend([Path(cr) for cr in candidate_roots])

    for p in items_to_check:
        if not p.exists() or not p.is_dir():
            continue
        if p.name.lower() == "datasets":
            for u_dir in p.iterdir():
                if u_dir.is_dir():
                    for ds_dir in u_dir.iterdir():
                        if ds_dir.is_dir():
                            add_root(ds_dir)
        else:
            add_root(p)

    if not discovered and target_base.exists() and target_base.is_dir():
        for child in target_base.iterdir():
            if not child.is_dir():
                continue
            if child.name.lower() == "datasets":
                for u_dir in child.iterdir():
                    if u_dir.is_dir():
                        for ds_dir in u_dir.iterdir():
                            if ds_dir.is_dir():
                                add_root(ds_dir)
            else:
                add_root(child)

    return discovered


@dataclass
class StagingReport:
    """Preflight diagnostic report of dataset discovery, duplicate resolution, and staging."""

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
    video_count_before: int = 0
    sensor_count_before: int = 0
    video_subjects_before: List[str] = field(default_factory=list)
    new_subjects_added: List[str] = field(default_factory=list)
    new_videos_added: int = 0
    new_sensors_added: int = 0

    # Extended multi-dataset Kaggle audit metadata
    attached_dataset_roots: List[str] = field(default_factory=list)
    sensor_root: Optional[str] = None
    sensor_modality_paths: Dict[str, str] = field(default_factory=dict)
    split_dir: Optional[str] = None
    official_split_files: Dict[str, str] = field(default_factory=dict)
    split_file_line_counts: Dict[str, int] = field(default_factory=dict)
    video_source_dirs: Dict[str, List[str]] = field(default_factory=dict)
    video_count_per_subject: Dict[str, int] = field(default_factory=dict)
    sensor_count_per_modality_subject: Dict[str, Dict[str, int]] = field(default_factory=dict)
    duplicate_recordings_count: int = 0
    duplicate_conflicts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """
        Validity requires:
        - At least one valid video recording (video_count > 0).
        - Sensor data in all four modalities with sensor_count > 0.
        - All four expected official split files present and non-empty.
        - Zero duplicate conflicts.
        - Zero missing required paths.
        """
        if self.video_count <= 0 or self.sensor_count <= 0:
            return False
        if len(self.duplicate_conflicts) > 0:
            return False
        if len(self.missing_required_paths) > 0:
            return False

        # All four sensor modalities must be present
        for exp_mod in self.expected_modalities:
            if exp_mod not in self.discovered_modalities:
                return False

        # All four expected official split files must be present with line count > 0
        for sp_name in EXPECTED_OFFICIAL_SPLITS:
            if sp_name not in self.official_split_files:
                return False
            if sp_name not in self.split_file_line_counts:
                return False
            if self.split_file_line_counts[sp_name] <= 0:
                return False

        return True

    def summary(self) -> str:
        lines = [
            "=" * 78,
            " DATASET PREFLIGHT DISCOVERY & STAGING REPORT",
            "=" * 78,
            f"Attached Dataset Roots  : {len(self.attached_dataset_roots)} detected",
        ]
        for r in self.attached_dataset_roots:
            lines.append(f"  * {r}")
        lines.extend([
            f"Canonical Staging Path : {self.canonical_dir}",
            f"Sensor Root Discovered : {self.sensor_root or 'None'}",
            f"Official Splits Path   : {self.split_dir or 'None'}",
            f"Existing Data Reused   : {self.reused}",
            f"Staging Duration       : {self.staging_duration_sec:.3f} seconds",
            f"Links / Copies Created : {self.linked_count} linked, {self.copied_count} copied, {self.repaired_count} repaired",
            "-" * 78,
            f"Video Subjects Staged  : {len(self.video_subjects)} subjects {sorted(self.video_subjects)}",
            f"Discovered Modalities  : {len(self.discovered_modalities)}/{len(self.expected_modalities)} {sorted(self.discovered_modalities)}",
            f"Video Files Count      : {self.video_count}",
            f"Sensor CSV Files Count : {self.sensor_count}",
            f"Official Split Files   : {self.split_count} files",
        ])
        for sf, lcnt in self.split_file_line_counts.items():
            lines.append(f"  - {sf:<26}: {lcnt} lines")

        if self.duplicate_recordings_count > 0 or self.duplicate_conflicts:
            lines.extend([
                "-" * 78,
                f"DUPLICATE SUBJECT RESOLUTION (e.g. subjects 8 and 16):",
                f"  Identical Duplicates Resolved: {self.duplicate_recordings_count} (reused single canonical copy)",
                f"  Duplicate Conflicts Detected : {len(self.duplicate_conflicts)}",
            ])
            for c in self.duplicate_conflicts[:5]:
                lines.append(f"    ! [CONFLICT] {c.get('recording_key')}: {c.get('sources')}")

        if self.video_subjects_before or self.new_subjects_added or self.new_videos_added > 0:
            lines.extend([
                "-" * 78,
                "DATASET EXPANSION (BEFORE VS AFTER STAGING):",
                f"  Initial Subjects Staged  : {sorted(self.video_subjects_before)} ({self.video_count_before} videos)",
                f"  Final Subjects Staged    : {sorted(self.video_subjects)} ({self.video_count} videos)",
                f"  New Subjects Discovered  : {self.new_subjects_added}",
                f"  Net Usable Videos Added  : +{self.new_videos_added}",
                f"  Net Usable Sensors Added : +{self.new_sensors_added}",
            ])
        if self.missing_required_paths:
            lines.append("-" * 78)
            lines.append("MISSING REQUIRED PATHS:")
            for p in self.missing_required_paths:
                lines.append(f"  - [MISSING] {p}")
        else:
            lines.append("-" * 78)
            lines.append("STATUS: All required MMAct modalities, splits, and video files verified.")
        lines.append("=" * 78)
        return "\n".join(lines)

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "canonical_dir": str(self.canonical_dir),
            "attached_dataset_roots": self.attached_dataset_roots,
            "sensor_root": self.sensor_root,
            "sensor_modalities": self.sensor_modality_paths,
            "official_splits_dir": self.split_dir,
            "official_split_files": self.official_split_files,
            "split_file_line_counts": self.split_file_line_counts,
            "video_source_dirs": self.video_source_dirs,
            "subjects_detected": sorted(self.video_subjects),
            "total_video_count": self.video_count,
            "total_sensor_count": self.sensor_count,
            "total_split_count": self.split_count,
            "video_count_per_subject": self.video_count_per_subject,
            "sensor_count_per_modality_subject": self.sensor_count_per_modality_subject,
            "duplicate_recordings_count": self.duplicate_recordings_count,
            "duplicate_conflicts": self.duplicate_conflicts,
            "missing_required_paths": self.missing_required_paths,
            "video_count_before": self.video_count_before,
            "video_count_after": self.video_count,
            "new_videos_added": self.new_videos_added,
            "sensor_count_before": self.sensor_count_before,
            "sensor_count_after": self.sensor_count,
            "new_sensors_added": self.new_sensors_added,
            "video_subjects_before": sorted(self.video_subjects_before),
            "video_subjects_after": sorted(self.video_subjects),
            "new_subjects_added": self.new_subjects_added,
            "staging_duration_sec": round(self.staging_duration_sec, 3),
            "reused_existing": self.reused,
        }


class FastMMActStager:
    """Discovers and stages MMAct dataset files using shallow, idempotent inspection."""

    EXPECTED_MODALITIES = EXPECTED_MODALITIES
    EXPECTED_OFFICIAL_SPLITS = EXPECTED_OFFICIAL_SPLITS

    def __init__(
        self,
        canonical_dir: Path,
        input_roots: Optional[List[Path]] = None,
        max_scan_depth: int = 4,
    ):
        self.canonical_dir = Path(canonical_dir)
        self.input_roots = input_roots
        self.max_scan_depth = max_scan_depth

        self.canonical_video = self.canonical_dir / "trimmed" / "video"
        self.canonical_sensors = self.canonical_dir / "trimmed" / "sensors"
        self.canonical_splits = self.canonical_dir / "trimmed" / "splits"

    @staticmethod
    def check_symlink_support(target_dir: Path) -> bool:
        """Tests whether directory symlinks can be created on the current filesystem."""
        target_dir.mkdir(parents=True, exist_ok=True)
        test_link = target_dir / ".test_symlink_probe"
        test_target = target_dir / ".test_symlink_target"
        try:
            test_target.touch()
            if test_link.exists() or test_link.is_symlink():
                test_link.unlink(missing_ok=True)
            os.symlink(test_target, test_link)
            supported = test_link.exists()
            test_link.unlink(missing_ok=True)
            test_target.unlink(missing_ok=True)
            return supported
        except (OSError, NotImplementedError):
            test_target.unlink(missing_ok=True)
            return False

    @staticmethod
    def safe_link_or_copy(src: Path, dst: Path) -> str:
        """
        Idempotently links or copies src to dst.
        - Never reuses a regular file based only on equal file size; verifies full-file SHA-256.
        - Never reuses a non-empty directory merely because it contains files; verifies expected contents.
        - Safely repairs stale or incomplete canonical data.
        - Never modifies the original read-only input datasets.
        Returns: 'reused', 'linked', 'copied', or 'repaired'
        """
        src_resolved = src.resolve()

        if src.is_file():
            if dst.is_symlink():
                try:
                    current_target = Path(os.path.realpath(dst)).resolve()
                    if current_target == src_resolved and dst.exists():
                        return "reused"
                    dst.unlink(missing_ok=True)
                    action = "repaired"
                except OSError:
                    dst.unlink(missing_ok=True)
                    action = "repaired"
            elif dst.is_file():
                try:
                    if dst.stat().st_size == src_resolved.stat().st_size and full_file_hash(dst) == full_file_hash(src_resolved):
                        return "reused"
                    dst.unlink(missing_ok=True)
                    action = "repaired"
                except OSError:
                    dst.unlink(missing_ok=True)
                    action = "repaired"
            elif dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink(missing_ok=True)
                action = "repaired"
            else:
                action = "linked"

            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(src_resolved, dst, target_is_directory=False)
                return action if action == "repaired" else "linked"
            except (OSError, NotImplementedError):
                shutil.copy2(src_resolved, dst)
                return action if action == "repaired" else "copied"

        elif src.is_dir():
            if dst.is_symlink():
                try:
                    current_target = Path(os.path.realpath(dst)).resolve()
                    if current_target == src_resolved and dst.exists():
                        return "reused"
                    dst.unlink(missing_ok=True)
                    action = "repaired"
                except OSError:
                    dst.unlink(missing_ok=True)
                    action = "repaired"
            elif dst.is_dir():
                src_files_set = {p.relative_to(src_resolved) for p in src_resolved.rglob("*") if p.is_file()}
                dst_files_set = {p.relative_to(dst) for p in dst.rglob("*") if p.is_file()}

                # Exact directory inventory comparison
                if src_files_set == dst_files_set:
                    all_match = True
                    for rel in src_files_set:
                        src_f = src_resolved / rel
                        dst_f = dst / rel
                        if dst_f.stat().st_size != src_f.stat().st_size or full_file_hash(dst_f) != full_file_hash(src_f):
                            all_match = False
                            break
                    if all_match:
                        return "reused"

                # Repair destination to match source exactly
                # 1. Remove extra stale files from dst
                extra_files = dst_files_set - src_files_set
                for extra_rel in extra_files:
                    extra_p = dst / extra_rel
                    extra_p.unlink(missing_ok=True)

                # Clean up any empty subdirectories in dst
                for root_dir, dirs, files in os.walk(dst, topdown=False):
                    if not dirs and not files and Path(root_dir) != dst:
                        try:
                            os.rmdir(root_dir)
                        except OSError:
                            pass

                # 2. Sync all expected source files to dst
                dst.parent.mkdir(parents=True, exist_ok=True)
                for rel in src_files_set:
                    src_f = src_resolved / rel
                    dst_f = dst / rel
                    FastMMActStager.safe_link_or_copy(src_f, dst_f)

                return "repaired"
            elif dst.exists():
                dst.unlink(missing_ok=True)
                action = "repaired"
            else:
                action = "linked"

            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(src_resolved, dst, target_is_directory=True)
                return action if action == "repaired" else "linked"
            except (OSError, NotImplementedError):
                shutil.copytree(src_resolved, dst, dirs_exist_ok=True)
                return action if action == "repaired" else "copied"

        return "reused"

    def check_existing_canonical(self) -> Tuple[bool, int, int, int, List[str], List[str]]:
        """
        Quickly checks if canonical_dir is already properly populated and valid.
        Requires:
        - At least 1 video file in canonical_video / cross_scene / trainval
        - At least 1 sensor file in each of the 4 expected sensor modalities
        - All 4 expected official split files present and non-empty
        """
        if not self.canonical_video.exists() or not self.canonical_sensors.exists() or not self.canonical_splits.exists():
            return False, 0, 0, 0, [], []

        video_subjects = []
        video_count = 0
        trainval_dir = self.canonical_video / "cross_scene" / "trainval"
        if trainval_dir.exists():
            for s_child in trainval_dir.iterdir():
                if s_child.is_dir():
                    norm_s = MMActIndexer.normalize_identifier(s_child.name, "subject")
                    video_subjects.append(norm_s)
                    for mp4 in s_child.glob("*/*/*/*.mp4"):
                        video_count += 1

        discovered_mods = []
        sensor_count = 0
        if self.canonical_sensors.exists():
            for exp_mod in self.EXPECTED_MODALITIES:
                mod_dir = self.canonical_sensors / exp_mod
                if mod_dir.exists() and mod_dir.is_dir():
                    mod_csvs = list(mod_dir.glob("*/*/*/*.csv")) or list(mod_dir.glob("*.csv"))
                    if len(mod_csvs) > 0:
                        discovered_mods.append(exp_mod)
                        sensor_count += len(mod_csvs)

        splits_present = []
        if self.canonical_splits.exists():
            for sp_name in self.EXPECTED_OFFICIAL_SPLITS:
                sp_file = self.canonical_splits / sp_name
                if sp_file.is_file() and sp_file.stat().st_size > 0:
                    splits_present.append(sp_name)

        split_count = len(splits_present)
        all_mods = set(self.EXPECTED_MODALITIES).issubset(set(discovered_mods))
        all_splits = set(self.EXPECTED_OFFICIAL_SPLITS).issubset(set(splits_present))

        is_valid = (
            video_count > 0
            and sensor_count > 0
            and len(video_subjects) > 0
            and all_mods
            and all_splits
        )
        return is_valid, video_count, sensor_count, split_count, sorted(list(set(video_subjects))), discovered_mods

    def discover_and_stage(self) -> StagingReport:
        """Executes bounded discovery, merged inventory duplicate resolution, and canonical staging."""
        t0 = time.time()

        # 1. Inspect existing canonical state (Before)
        is_valid_before, v_cnt_before, s_cnt_before, sp_cnt_before, v_subjs_before, d_mods_before = self.check_existing_canonical()

        # 2. Discover all attached dataset roots (e.g. /kaggle/input/datasets/*/* or /kaggle/input/*)
        if self.input_roots is not None:
            roots = discover_kaggle_input_roots(self.input_roots)
        else:
            roots = discover_kaggle_input_roots()
            if not roots:
                local_data = Path("data")
                if local_data.exists():
                    roots.append(local_data)

        attached_roots_str = [str(r) for r in roots]

        # Early return if no roots and existing canonical data is already valid
        if not roots and is_valid_before:
            dur = time.time() - t0
            sp_line_counts = {}
            official_splits = {}
            for sp_name in self.EXPECTED_OFFICIAL_SPLITS:
                sp_f = self.canonical_splits / sp_name
                if sp_f.exists():
                    official_splits[sp_name] = str(sp_f)
                    try:
                        sp_line_counts[sp_name] = sum(1 for ln in sp_f.read_text(encoding="utf-8").splitlines() if ln.strip())
                    except Exception:
                        sp_line_counts[sp_name] = 0

            return StagingReport(
                dataset_root_selected=self.canonical_dir,
                canonical_dir=self.canonical_dir,
                expected_modalities=self.EXPECTED_MODALITIES,
                discovered_modalities=d_mods_before,
                video_count=v_cnt_before,
                sensor_count=s_cnt_before,
                split_count=sp_cnt_before,
                video_subjects=v_subjs_before,
                staging_duration_sec=dur,
                reused=True,
                video_count_before=v_cnt_before,
                sensor_count_before=s_cnt_before,
                video_subjects_before=v_subjs_before,
                attached_dataset_roots=attached_roots_str,
                split_dir=str(self.canonical_splits),
                official_split_files=official_splits,
                split_file_line_counts=sp_line_counts,
                sensor_root=str(self.canonical_sensors),
                sensor_modality_paths={m: str(self.canonical_sensors / m) for m in d_mods_before},
            )

        if not roots and not is_valid_before:
            dur = time.time() - t0
            return StagingReport(
                dataset_root_selected=self.canonical_dir,
                canonical_dir=self.canonical_dir,
                expected_modalities=self.EXPECTED_MODALITIES,
                discovered_modalities=[],
                missing_required_paths=["Existing canonical data is incomplete or invalid, and no input dataset roots found in /kaggle/input or local storage."],
                staging_duration_sec=dur,
                reused=False,
            )

        # 3. Discovery Data Structures
        discovered_sensor_root: Optional[Path] = None
        discovered_sensors: Dict[str, Path] = {}  # canon_mod -> Path
        sensor_csv_counts: Dict[str, Dict[str, int]] = {}  # mod -> {subj: count}

        discovered_split_dir: Optional[Path] = None
        discovered_splits: Dict[str, Path] = {}
        split_line_counts: Dict[str, int] = {}

        discovered_recordings: Dict[Tuple[str, str, str, str, str], List[Path]] = {}
        discovered_archives: List[Path] = []
        video_source_dirs_manifest: Dict[str, List[str]] = {}

        canon_resolved = self.canonical_dir.resolve()

        def is_under_canonical(p: Path) -> bool:
            try:
                p.resolve().relative_to(canon_resolved)
                return True
            except (ValueError, OSError):
                return False

        # 4. Targeted Discovery across roots
        for root in roots:
            # 4A. Search for Sensors: check nested trimmed_sensor/sensor, sensor, sensors
            sensor_candidate_dirs = [
                root / "trimmed_sensor" / "sensor",
                root / "trimmed_sensor",
                root / "sensor",
                root / "sensors",
                root / "trimmed" / "sensors",
                root / "trimmed" / "sensor",
            ]
            for sc_dir in sensor_candidate_dirs:
                if not sc_dir.exists() or not sc_dir.is_dir() or is_under_canonical(sc_dir):
                    continue
                try:
                    for child in sc_dir.iterdir():
                        if not child.is_dir():
                            continue
                        c_lower = child.name.lower()
                        if c_lower in MMActIndexer.SENSOR_NAME_MAP:
                            raw_mod = MMActIndexer.SENSOR_NAME_MAP[c_lower]
                            canon_mod = raw_mod if raw_mod.endswith("_clip") else f"{raw_mod}_clip"
                            if canon_mod not in discovered_sensors:
                                discovered_sensors[canon_mod] = child
                                if discovered_sensor_root is None:
                                    discovered_sensor_root = sc_dir
                                sensor_csv_counts.setdefault(canon_mod, {})
                                for s_sub in child.iterdir():
                                    if s_sub.is_dir():
                                        s_m = re.search(r"subject_?(\d+)", s_sub.name.lower())
                                        if s_m:
                                            norm_s = f"subject{int(s_m.group(1))}"
                                            csv_cnt = len(list(s_sub.glob("*/*/*.csv"))) or len(list(s_sub.glob("*.csv")))
                                            sensor_csv_counts[canon_mod][norm_s] = csv_cnt
                except (PermissionError, OSError):
                    pass

            # 4B. Search for Official Splits: trimmed_splits/splits, splits, trimmed/splits
            split_candidate_dirs = [
                root / "trimmed_splits" / "splits",
                root / "splits",
                root / "trimmed" / "splits",
            ]
            for sp_dir in split_candidate_dirs:
                if not sp_dir.exists() or not sp_dir.is_dir() or is_under_canonical(sp_dir):
                    continue
                try:
                    for f in sp_dir.iterdir():
                        if f.is_file() and f.suffix.lower() == ".txt":
                            name_lower = f.name.lower()
                            if any(k in name_lower for k in ["cross_scene", "cross_view", "train", "val", "test"]):
                                if f.name not in discovered_splits:
                                    discovered_splits[f.name] = f
                                    if discovered_split_dir is None:
                                        discovered_split_dir = sp_dir
                                    try:
                                        lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
                                        split_line_counts[f.name] = len(lines)
                                    except Exception:
                                        split_line_counts[f.name] = 0
                except (PermissionError, OSError):
                    pass

            # 4C. Search for Archives and Video Recordings
            # Check for archive files
            for arc_cand in root.glob("*"):
                if arc_cand.is_file() and arc_cand.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
                    discovered_archives.append(arc_cand)

            # Discover all MP4 files under this root
            all_mp4s = MMActIndexer.find_files_recursively(root, {".mp4"})
            for mp4 in all_mp4s:
                if is_under_canonical(mp4):
                    continue
                parsed = MMActIndexer.parse_video_path(mp4)
                if parsed is not None:
                    norm_s = parsed[0]
                    discovered_recordings.setdefault(parsed, []).append(mp4)
                    video_source_dirs_manifest.setdefault(norm_s, [])
                    p_parent = str(mp4.parent)
                    if p_parent not in video_source_dirs_manifest[norm_s]:
                        video_source_dirs_manifest[norm_s].append(p_parent)

        # 5. Extract archives if any are detected
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

        # 6. Duplicate Detection & Merged Recording Inventory
        identical_dup_count = 0
        duplicate_conflicts: List[Dict[str, Any]] = []
        canonical_recordings: Dict[Tuple[str, str, str, str, str], Path] = {}
        video_count_per_subject: Dict[str, int] = {}

        for rec_key, src_paths in discovered_recordings.items():
            norm_s = rec_key[0]
            # Deduplicate references to the exact same path on disk
            unique_src_paths = []
            seen_res_paths = set()
            for p in src_paths:
                try:
                    res_p = str(p.resolve())
                except OSError:
                    res_p = str(p)
                if res_p not in seen_res_paths:
                    seen_res_paths.add(res_p)
                    unique_src_paths.append(p)

            if len(unique_src_paths) == 1:
                canonical_recordings[rec_key] = unique_src_paths[0]
                video_count_per_subject[norm_s] = video_count_per_subject.get(norm_s, 0) + 1
            else:
                # Compare EVERY source containing this recording key
                files = unique_src_paths
                has_conflict = False
                sizes = []
                hashes = []
                conflict_reasons = []

                for f in files:
                    try:
                        if not f.exists() or not f.is_file():
                            has_conflict = True
                            conflict_reasons.append(f"File missing or not a regular file: {f}")
                            break
                        sz = f.stat().st_size
                        h = full_file_hash(f)
                        sizes.append(sz)
                        hashes.append(h)
                    except (OSError, PermissionError) as e:
                        has_conflict = True
                        conflict_reasons.append(f"Unreadable file {f}: {e}")
                        break

                if not has_conflict:
                    ref_sz = sizes[0]
                    ref_h = hashes[0]
                    for idx in range(1, len(files)):
                        if sizes[idx] != ref_sz or hashes[idx] != ref_h:
                            has_conflict = True
                            conflict_reasons.append(
                                f"Byte discrepancy between source 0 ({files[0]}, size={ref_sz}, hash={ref_h[:12]}...) "
                                f"and source {idx} ({files[idx]}, size={sizes[idx]}, hash={hashes[idx][:12]}...)"
                            )

                if has_conflict:
                    duplicate_conflicts.append({
                        "subject": norm_s,
                        "recording_key": str(rec_key),
                        "sources": [str(f) for f in files],
                        "reasons": conflict_reasons,
                    })
                else:
                    # All copies are byte-identical across all sources
                    identical_dup_count += (len(files) - 1)
                    # Prefer expanded cross_scene_video path if available
                    chosen = next((f for f in files if "cross_scene_video" in str(f)), files[0])
                    canonical_recordings[rec_key] = chosen
                    video_count_per_subject[norm_s] = video_count_per_subject.get(norm_s, 0) + 1

        # 7. Canonical Staging Execution
        linked_cnt, copied_cnt, repaired_cnt = 0, 0, 0
        canonical_trainval = self.canonical_video / "cross_scene" / "trainval"
        canonical_trainval.mkdir(parents=True, exist_ok=True)

        # Stage each recording from the merged inventory into canonical storage
        for rec_key, src_file in canonical_recordings.items():
            subj, cam, scene, session, action = rec_key
            dst_file = canonical_trainval / subj / cam / scene / session / f"{action}.mp4"
            stat = self.safe_link_or_copy(src_file, dst_file)
            if stat == "linked":
                linked_cnt += 1
            elif stat == "copied":
                copied_cnt += 1
            elif stat == "repaired":
                repaired_cnt += 1

        # Stage Sensors
        self.canonical_sensors.mkdir(parents=True, exist_ok=True)
        for canon_mod, src_dir in discovered_sensors.items():
            dst_mod = self.canonical_sensors / canon_mod
            stat = self.safe_link_or_copy(src_dir, dst_mod)
            if stat == "linked":
                linked_cnt += 1
            elif stat == "copied":
                copied_cnt += 1
            elif stat == "repaired":
                repaired_cnt += 1

        # Also alias trimmed/sensor -> trimmed/sensors for tools expecting singular
        sensor_alias = self.canonical_dir / "trimmed" / "sensor"
        if not sensor_alias.exists() or sensor_alias.is_symlink():
            self.safe_link_or_copy(self.canonical_sensors, sensor_alias)

        # Stage Splits
        self.canonical_splits.mkdir(parents=True, exist_ok=True)
        for sp_name, src_file in discovered_splits.items():
            dst_split = self.canonical_splits / sp_name
            stat = self.safe_link_or_copy(src_file, dst_split)
            if stat == "linked":
                linked_cnt += 1
            elif stat == "copied":
                copied_cnt += 1
            elif stat == "repaired":
                repaired_cnt += 1

        # 8. Post-Staging Count Verification
        is_valid_after, v_cnt_after, s_cnt_after, sp_cnt_after, v_subjs_after, d_mods_after = self.check_existing_canonical()

        dur = time.time() - t0
        missing_required = []
        if v_cnt_after == 0:
            missing_required.append("No video files found after staging.")
        if s_cnt_after == 0:
            missing_required.append("No sensor files found after staging.")
        for exp_mod in self.EXPECTED_MODALITIES:
            if exp_mod not in d_mods_after:
                missing_required.append(f"Missing required sensor modality: {exp_mod}")
        for sp_name in self.EXPECTED_OFFICIAL_SPLITS:
            if sp_name not in discovered_splits or not (self.canonical_splits / sp_name).exists():
                missing_required.append(f"Missing required official split file: {sp_name}")
            elif (self.canonical_splits / sp_name).stat().st_size == 0:
                missing_required.append(f"Official split file is empty: {sp_name}")
        if duplicate_conflicts:
            missing_required.append(f"{len(duplicate_conflicts)} duplicate recording conflicts detected across sources.")

        new_vids = max(0, v_cnt_after - v_cnt_before)
        new_sens = max(0, s_cnt_after - s_cnt_before)
        new_subjs = sorted(list(set(v_subjs_after) - set(v_subjs_before)))
        is_reused = (
            is_valid_before
            and is_valid_after
            and (new_vids == 0)
            and (new_sens == 0)
            and (len(new_subjs) == 0)
            and (linked_cnt == 0)
            and (copied_cnt == 0)
            and (repaired_cnt == 0)
            and (len(duplicate_conflicts) == 0)
        )

        report = StagingReport(
            dataset_root_selected=self.canonical_dir,
            canonical_dir=self.canonical_dir,
            expected_modalities=self.EXPECTED_MODALITIES,
            discovered_modalities=d_mods_after,
            video_count=v_cnt_after,
            sensor_count=s_cnt_after,
            split_count=sp_cnt_after,
            video_subjects=v_subjs_after,
            missing_required_paths=missing_required,
            staging_duration_sec=dur,
            reused=is_reused,
            linked_count=linked_cnt,
            copied_count=copied_cnt,
            repaired_count=repaired_cnt,
            video_count_before=v_cnt_before,
            sensor_count_before=s_cnt_before,
            video_subjects_before=v_subjs_before,
            new_subjects_added=new_subjs,
            new_videos_added=new_vids,
            new_sensors_added=new_sens,
            attached_dataset_roots=attached_roots_str,
            sensor_root=str(discovered_sensor_root) if discovered_sensor_root else None,
            sensor_modality_paths={m: str(p) for m, p in discovered_sensors.items()},
            split_dir=str(discovered_split_dir) if discovered_split_dir else None,
            official_split_files={n: str(p) for n, p in discovered_splits.items()},
            split_file_line_counts=split_line_counts,
            video_source_dirs=video_source_dirs_manifest,
            video_count_per_subject=video_count_per_subject,
            sensor_count_per_modality_subject=sensor_csv_counts,
            duplicate_recordings_count=identical_dup_count,
            duplicate_conflicts=duplicate_conflicts,
        )

        # 9. Save machine-readable manifest JSON
        manifest_data = report.to_manifest_dict()
        try:
            with open(self.canonical_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            for m_dir in [
                self.canonical_dir / "manifests",
                Path("outputs/manifests"),
                self.canonical_dir.parent.parent / "outputs" / "manifests",
            ]:
                try:
                    m_dir.mkdir(parents=True, exist_ok=True)
                    with open(m_dir / "dataset_expansion_manifest.json", "w", encoding="utf-8") as f:
                        json.dump(manifest_data, f, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

        return report
