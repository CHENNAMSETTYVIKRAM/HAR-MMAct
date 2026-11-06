"""
splits.py - Leakage-Free Dataset Splitting & Strict Official Split Protocols

Enforces strict separation between training, validation, and test sets.
Supports:
1. Official MMAct Protocols:
   - 'official_x_scene' (trimmed_x-scene_train.txt, trimmed_x-scene_val.txt)
   - 'official_x_view'  (trimmed_x-view_train.txt, trimmed_x-view_val.txt)
   Strictly adheres to official source split files.
   Never silently falls back to rule-based splits.
   Test set is empty by default (official release contains only train & validation).
   Custom validation-to-test partitioning is opt-in and explicitly documented.
2. Rule-Based Protocols:
   - 'subject' (subject-independent)
   - 'cross_scene' (scene-independent: scenes 1,2 train / scenes 3,4 val)
   - 'cross_view' (view-independent: cams 1,2,4 train / cam 3 val)
   - 'session' (recording session-independent)
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.config import PROJECT_ROOT, SplitConfig, VERIFIED_TARGET_CLASSES
from src.data.indexer import MMActIndexer, PairedSample


@dataclass
class SplitAuditReport:
    """Detailed audit report of dataset splitting, recording membership, and anomalies."""

    strategy: str
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    train_recordings: int = 0
    val_recordings: int = 0
    test_recordings: int = 0
    is_official: bool = False
    is_custom_test: bool = False
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    duplicate_train_entries: List[str] = field(default_factory=list)
    duplicate_val_entries: List[str] = field(default_factory=list)
    overlapping_split_entries: List[str] = field(default_factory=list)
    malformed_split_entries: List[str] = field(default_factory=list)
    unmatched_indexed_samples: List[str] = field(default_factory=list)
    unmatched_split_recordings: List[str] = field(default_factory=list)
    partition_reason: Optional[str] = None

    def summary(self) -> str:
        lines = [
            "=" * 78,
            " SPLIT AUDIT & INTEGRITY REPORT",
            "=" * 78,
            f"Strategy                : {self.strategy}",
            f"Official Split Protocol : {self.is_official}",
            f"Custom Test Partition   : {self.is_custom_test}",
            f"Train Source File       : {self.train_file or 'N/A (Rule-based)'}",
            f"Val Source File         : {self.val_file or 'N/A (Rule-based)'}",
            f"Train Samples / Records : {self.train_count} samples ({self.train_recordings} recordings)",
            f"Val Samples / Records   : {self.val_count} samples ({self.val_recordings} recordings)",
            f"Test Samples / Records  : {self.test_count} samples ({self.test_recordings} recordings)",
            "-" * 78,
            f"Duplicate Train Entries : {len(self.duplicate_train_entries)}",
            f"Duplicate Val Entries   : {len(self.duplicate_val_entries)}",
            f"Overlapping Split Entr. : {len(self.overlapping_split_entries)}",
            f"Malformed Split Entries : {len(self.malformed_split_entries)}",
            f"Unmatched Indexed Smpls : {len(self.unmatched_indexed_samples)}",
            f"Unmatched Split Records : {len(self.unmatched_split_recordings)}",
        ]
        if self.partition_reason:
            lines.append(f"Partition Note          : {self.partition_reason}")
        lines.append("=" * 78)
        return "\n".join(lines)


def find_official_split_files(
    mode: str,
    dataset_root: Optional[Union[str, Path]] = None,
    split_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """
    Locates official MMAct split files for 'scene' or 'view'.
    Searches across documented canonical and configured directories without guessing.
    Raises FileNotFoundError if official files cannot be resolved (no silent fallback).
    """
    prefix = f"trimmed_x-{mode}"
    train_name = f"{prefix}_train.txt"
    val_name = f"{prefix}_val.txt"

    candidate_dirs: List[Path] = []

    if split_dir:
        candidate_dirs = [Path(split_dir)]
    else:
        if dataset_root:
            d_root = Path(dataset_root)
            candidate_dirs.extend([
                d_root / "trimmed" / "splits",
                d_root / "splits",
                d_root / "trimmed_splits" / "splits",
                d_root / "trimmed_splits",
                d_root,
            ])

        # Standard repository layout paths
        candidate_dirs.extend([
            PROJECT_ROOT / "data" / "genuine_mmact" / "trimmed" / "splits",
            PROJECT_ROOT / "data" / "downloads" / "trimmed_splits" / "splits",
            PROJECT_ROOT / "data" / "splits",
            Path("data/genuine_mmact/trimmed/splits"),
            Path("data/downloads/trimmed_splits/splits"),
            Path("data/splits"),
        ])

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_candidates: List[Path] = []
    for cd in candidate_dirs:
        cd_str = str(cd.resolve()) if cd.is_absolute() else str(cd)
        if cd_str not in seen:
            seen.add(cd_str)
            unique_candidates.append(cd)

    searched: List[str] = []
    for d in unique_candidates:
        searched.append(str(d))
        t_path = d / train_name
        v_path = d / val_name
        if t_path.exists() and v_path.exists():
            return t_path.resolve(), v_path.resolve()

    searched_str = "\n  - ".join(searched)
    raise FileNotFoundError(
        f"CRITICAL: Official MMAct split files for mode '{mode}' ({train_name}, {val_name}) "
        f"could not be found in any of the searched locations:\n  - {searched_str}\n"
        f"Please provide the valid split directory via SplitConfig.split_dir or place split files "
        f"in <dataset_root>/trimmed/splits/."
    )


class DatasetSplitter:
    """Partitions indexed samples into strictly isolated train, validation, and test sets."""

    def __init__(self, config: SplitConfig, dataset_root: Optional[Union[str, Path]] = None):
        self.config = config
        self.dataset_root = Path(dataset_root) if dataset_root else (
            Path(config.dataset_root) if config.dataset_root else None
        )
        self.last_audit: Optional[SplitAuditReport] = None

    def split(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """
        Splits samples according to config.split_strategy:
        - 'official_x_scene', 'official_scene': Official cross-scene split files.
        - 'official_x_view', 'official_view'  : Official cross-view split files.
        - 'subject'                           : Subject-independent rule split.
        - 'cross_scene', 'scene'              : Rule-based cross-scene split.
        - 'cross_view'                        : Rule-based cross-view split.
        - 'session'                           : Recording session-independent split.
        """
        if not samples:
            self.last_audit = SplitAuditReport(strategy=self.config.split_strategy)
            return ([], [], [])

        strategy = self.config.split_strategy.lower()

        if strategy in ["official_x_scene", "official_scene", "official_cross_scene"]:
            return self._split_by_official_records(samples, mode="scene")
        elif strategy in ["official_x_view", "official_view", "official_cross_view"]:
            return self._split_by_official_records(samples, mode="view")
        elif strategy in ["cross_view", "view"]:
            return self._split_by_cross_view(samples)
        elif strategy in ["scene", "cross_scene"]:
            return self._split_by_cross_scene(samples)
        elif strategy == "session":
            return self._split_by_session(samples)
        else:  # default 'subject'
            return self._split_by_subject(samples)

    @staticmethod
    def parse_split_line(line: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Parses an official split record line and extracts (subject, cam, scene, session).
        Expected pattern: cross_(scene|view)/trainval/<subject>/<cam>/<scene>/<session>
        Accepts normalized or non-normalized variations, Windows or POSIX separators.
        """
        clean = line.replace("\\", "/").strip().strip("/")
        if not clean:
            return None

        # Match pattern containing subject, cam, scene, session
        match = re.search(
            r"(?:cross_(?:scene|view)/trainval/)?(subject_?\d+|sub_?\d+)/(cam_?\d+|camera_?\d+|view_?\d+|ego)/(scene_?\d+)/(session_?\d+)",
            clean,
            re.IGNORECASE,
        )
        if match:
            subj_raw, cam_raw, scene_raw, sess_raw = match.groups()
            subj = MMActIndexer.normalize_identifier(subj_raw, "subject")
            cam = "ego" if cam_raw.lower() == "ego" else MMActIndexer.normalize_identifier(cam_raw, "cam")
            scene = MMActIndexer.normalize_identifier(scene_raw, "scene")
            session = MMActIndexer.normalize_identifier(sess_raw, "session")
            return (subj, cam, scene, session)
        return None

    def _split_by_official_records(
        self, samples: List[PairedSample], mode: str = "scene"
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """
        Partitions samples using official MMAct challenge split definition files.
        Never silently falls back.
        Detects duplicates, overlapping entries, unmatched samples, and malformed lines.
        """
        train_file, val_file = find_official_split_files(
            mode=mode,
            dataset_root=self.dataset_root,
            split_dir=self.config.split_dir,
        )

        audit = SplitAuditReport(
            strategy=f"official_x_{mode}",
            is_official=True,
            train_file=str(train_file),
            val_file=str(val_file),
        )

        # 1. Parse train split file
        train_records: Set[Tuple[str, str, str, str]] = set()
        seen_train: Set[str] = set()
        with open(train_file, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f, 1):
                clean_line = raw_line.strip()
                if not clean_line:
                    continue
                parsed = self.parse_split_line(clean_line)
                if not parsed:
                    audit.malformed_split_entries.append(f"train:{line_idx}:{clean_line}")
                    continue
                canonical_id = f"{parsed[0]}/{parsed[1]}/{parsed[2]}/{parsed[3]}"
                if canonical_id in seen_train:
                    audit.duplicate_train_entries.append(f"train:{line_idx}:{canonical_id}")
                seen_train.add(canonical_id)
                train_records.add(parsed)

        # 2. Parse val split file
        val_records: Set[Tuple[str, str, str, str]] = set()
        seen_val: Set[str] = set()
        with open(val_file, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f, 1):
                clean_line = raw_line.strip()
                if not clean_line:
                    continue
                parsed = self.parse_split_line(clean_line)
                if not parsed:
                    audit.malformed_split_entries.append(f"val:{line_idx}:{clean_line}")
                    continue
                canonical_id = f"{parsed[0]}/{parsed[1]}/{parsed[2]}/{parsed[3]}"
                if canonical_id in seen_val:
                    audit.duplicate_val_entries.append(f"val:{line_idx}:{canonical_id}")
                seen_val.add(canonical_id)
                val_records.add(parsed)

        # 3. Detect train/val recording overlap
        overlap = train_records & val_records
        if overlap:
            for ov in overlap:
                audit.overlapping_split_entries.append(f"{ov[0]}/{ov[1]}/{ov[2]}/{ov[3]}")
            raise ValueError(
                f"CRITICAL: Official split files contain {len(overlap)} overlapping recording IDs "
                f"between train ({train_file.name}) and val ({val_file.name})! Example overlap: {list(overlap)[:3]}"
            )

        # 4. Map indexed samples to official recordings
        train_samples: List[PairedSample] = []
        val_all: List[PairedSample] = []
        matched_train_records: Set[Tuple[str, str, str, str]] = set()
        matched_val_records: Set[Tuple[str, str, str, str]] = set()

        for s in samples:
            key = (s.subject, s.cam, s.scene, s.session)
            if key in train_records:
                train_samples.append(s)
                matched_train_records.add(key)
            elif key in val_records:
                val_all.append(s)
                matched_val_records.add(key)
            else:
                audit.unmatched_indexed_samples.append(s.sample_id)

        # Record split entries with zero matching indexed samples
        for tr in train_records:
            if tr not in matched_train_records:
                audit.unmatched_split_recordings.append(f"train:{tr}")
        for vr in val_records:
            if vr not in matched_val_records:
                audit.unmatched_split_recordings.append(f"val:{vr}")

        # 5. Handle test partitioning (opt-in only)
        if getattr(self.config, "partition_val_for_test", False):
            audit.is_custom_test = True
            val_samples, test_samples, reason = self._partition_groups_class_aware(val_all)
            audit.partition_reason = reason
        else:
            audit.is_custom_test = False
            val_samples = val_all
            test_samples = []

        audit.train_count = len(train_samples)
        audit.val_count = len(val_samples)
        audit.test_count = len(test_samples)
        audit.train_recordings = len({(s.subject, s.cam, s.scene, s.session) for s in train_samples})
        audit.val_recordings = len({(s.subject, s.cam, s.scene, s.session) for s in val_samples})
        audit.test_recordings = len({(s.subject, s.cam, s.scene, s.session) for s in test_samples})
        self.last_audit = audit

        return train_samples, val_samples, test_samples

    def _partition_groups_class_aware(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], str]:
        """
        Partitions evaluation samples at the atomic recording group level (subject, scene, session)
        using deterministic class-aware balancing.
        Guarantees that ALL cameras and actions from the same (subject, scene, session) stay
        strictly in one partition, preventing inter-camera recording leakage.
        """
        if not samples:
            return [], [], "No samples to partition."

        # Group by atomic recording session: (subject, scene, session)
        group_map: Dict[Tuple[str, str, str], List[PairedSample]] = {}
        for s in samples:
            g_key = (s.subject, s.scene, s.session)
            group_map.setdefault(g_key, []).append(s)

        group_keys = sorted(list(group_map.keys()))
        if len(group_keys) <= 1:
            return samples, [], "Only 1 recording session group available; cannot partition without intra-session leakage."

        seed = getattr(self.config, "seed", 42)
        rng = np.random.RandomState(seed)
        shuffled_keys = list(group_keys)
        rng.shuffle(shuffled_keys)

        # Deterministic greedy balancing for class and sample count parity
        val_samples: List[PairedSample] = []
        test_samples: List[PairedSample] = []
        val_class_counts: Dict[str, int] = {c: 0 for c in VERIFIED_TARGET_CLASSES}
        test_class_counts: Dict[str, int] = {c: 0 for c in VERIFIED_TARGET_CLASSES}

        for gk in shuffled_keys:
            g_samples = group_map[gk]
            g_class_counts = {}
            for s in g_samples:
                g_class_counts[s.action] = g_class_counts.get(s.action, 0) + 1

            # Compute assignment score: preference for the partition needing these classes more
            val_score = 0
            test_score = 0
            for act, cnt in g_class_counts.items():
                val_score += val_class_counts.get(act, 0)
                test_score += test_class_counts.get(act, 0)

            # Assign to partition with fewer samples of these actions (or fewer total samples)
            if val_score < test_score:
                assign_to_val = True
            elif test_score < val_score:
                assign_to_val = False
            else:
                assign_to_val = (len(val_samples) <= len(test_samples))

            if assign_to_val:
                val_samples.extend(g_samples)
                for act, cnt in g_class_counts.items():
                    val_class_counts[act] = val_class_counts.get(act, 0) + cnt
            else:
                test_samples.extend(g_samples)
                for act, cnt in g_class_counts.items():
                    test_class_counts[act] = test_class_counts.get(act, 0) + cnt

        # Verify zero group leakage
        val_g_set = {(s.subject, s.scene, s.session) for s in val_samples}
        test_g_set = {(s.subject, s.scene, s.session) for s in test_samples}
        assert len(val_g_set & test_g_set) == 0, "CRITICAL: Group overlap detected across val and test partitions!"

        missing_test_classes = [c for c in VERIFIED_TARGET_CLASSES if test_class_counts.get(c, 0) == 0]
        reason = (
            f"Class-aware atomic partition across {len(group_keys)} recording groups (seed={seed}). "
            f"Val: {len(val_samples)} samples ({len(val_g_set)} groups), "
            f"Test: {len(test_samples)} samples ({len(test_g_set)} groups)."
        )
        if missing_test_classes:
            reason += f" Classes present only in validation: {missing_test_classes} (each occurred in only 1 recording group)."

        return val_samples, test_samples, reason

    def _split_by_cross_view(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """
        Rule-Based MMAct Task 1 Cross-View Protocol:
        Held-out validation view is cam3. Train on cam1, cam2, cam4.
        """
        all_cams = sorted(list({s.cam for s in samples}))
        if "cam3" in all_cams:
            train = [s for s in samples if s.cam != "cam3"]
            val_all = [s for s in samples if s.cam == "cam3"]
        elif len(all_cams) >= 2:
            train = [s for s in samples if s.cam == all_cams[0]]
            val_all = [s for s in samples if s.cam != all_cams[0]]
        else:
            return self._split_by_session(samples)

        if getattr(self.config, "partition_val_for_test", False):
            val, test, reason = self._partition_groups_class_aware(val_all)
        else:
            val, test = val_all, []

        self._record_rule_audit("cross_view", train, val, test)
        return train, val, test

    def _split_by_cross_scene(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """
        Rule-Based MMAct Task 1 Cross-Scene Protocol:
        Train on scenes 1 and 2. Validation on scenes 3 and 4.
        """
        all_scenes = sorted(list({s.scene for s in samples}))
        has_eval_scenes = any(sc in ["scene3", "scene4"] for sc in all_scenes)
        if has_eval_scenes:
            train = [s for s in samples if s.scene in ["scene1", "scene2"]]
            val_all = [s for s in samples if s.scene in ["scene3", "scene4"]]
        elif len(all_scenes) >= 2:
            train = [s for s in samples if s.scene == all_scenes[0]]
            val_all = [s for s in samples if s.scene != all_scenes[0]]
        else:
            return self._split_by_session(samples)

        if getattr(self.config, "partition_val_for_test", False):
            val, test, reason = self._partition_groups_class_aware(val_all)
        else:
            val, test = val_all, []

        self._record_rule_audit("cross_scene", train, val, test)
        return train, val, test

    def _split_by_subject(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """Partitions by subject ID (strictly subject-independent)."""
        all_subjects = sorted(list({s.subject for s in samples}))

        configured_train = set(self.config.train_subjects) & set(all_subjects)
        configured_val = set(self.config.val_subjects) & set(all_subjects)
        configured_test = set(self.config.test_subjects) & set(all_subjects)

        if not configured_train or not configured_val:
            n_total = len(all_subjects)
            if n_total >= 3:
                test_set = {all_subjects[-1]}
                val_set = {all_subjects[-2]}
                train_set = set(all_subjects[:-2])
            elif n_total == 2:
                # 2 subjects: assign first subject to train, second subject to validation
                train_subj = all_subjects[0]
                eval_subj = all_subjects[1]
                train = [s for s in samples if s.subject == train_subj]
                eval_samples = [s for s in samples if s.subject == eval_subj]
                if getattr(self.config, "partition_val_for_test", False):
                    val, test, reason = self._partition_groups_class_aware(eval_samples)
                else:
                    val, test = eval_samples, []
                self._record_rule_audit("subject", train, val, test)
                return train, val, test
            else:
                return self._split_by_session(samples)
        else:
            train_set = configured_train
            val_set = configured_val - train_set
            test_set = configured_test - train_set - val_set

        train = [s for s in samples if s.subject in train_set]
        val = [s for s in samples if s.subject in val_set]
        test = [s for s in samples if s.subject in test_set]

        if len(test) == 0 and len(val) > 0 and getattr(self.config, "partition_val_for_test", False):
            val, test, reason = self._partition_groups_class_aware(val)

        self._record_rule_audit("subject", train, val, test)
        return train, val, test

    def _split_by_session(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """Partitions by unique recording session to avoid recording-level leakage."""
        sessions = sorted(list({(s.subject, s.scene, s.session) for s in samples}))
        seed = getattr(self.config, "seed", 42)
        rng = np.random.RandomState(seed)
        shuffled_sessions = list(sessions)
        rng.shuffle(shuffled_sessions)

        n_total = len(shuffled_sessions)
        n_train = max(1, int(round(n_total * self.config.train_ratio)))
        n_val = max(1, int(round(n_total * self.config.val_ratio)))

        train_sessions = set(shuffled_sessions[:n_train])
        val_sessions = set(shuffled_sessions[n_train : n_train + n_val])
        test_sessions = set(shuffled_sessions[n_train + n_val :])

        train = [s for s in samples if (s.subject, s.scene, s.session) in train_sessions]
        val = [s for s in samples if (s.subject, s.scene, s.session) in val_sessions]
        test = [s for s in samples if (s.subject, s.scene, s.session) in test_sessions]

        self._record_rule_audit("session", train, val, test)
        return train, val, test

    def _record_rule_audit(
        self, strategy: str, train: List[PairedSample], val: List[PairedSample], test: List[PairedSample]
    ):
        self.last_audit = SplitAuditReport(
            strategy=strategy,
            is_official=False,
            is_custom_test=len(test) > 0 and getattr(self.config, "partition_val_for_test", False),
            train_count=len(train),
            val_count=len(val),
            test_count=len(test),
            train_recordings=len({(s.subject, s.cam, s.scene, s.session) for s in train}),
            val_recordings=len({(s.subject, s.cam, s.scene, s.session) for s in val}),
            test_recordings=len({(s.subject, s.cam, s.scene, s.session) for s in test}),
        )


def verify_split_isolation(
    train: List[PairedSample],
    val: List[PairedSample],
    test: List[PairedSample],
    strategy: str = "subject",
) -> Dict[str, Any]:
    """
    Formally verifies that there is zero overlap between train, validation, and test splits.
    Checks:
    - video-path overlap
    - recording-ID overlap: (subject, cam, scene, session)
    - session/group overlap: (subject, scene, session)
    - subject overlap when strategy is 'subject'
    - camera overlap when strategy is 'cross_view' or 'official_x_view'
    - scene overlap when strategy is 'cross_scene' or 'official_x_scene'
    - pairwise isolation across train, val, test
    Does NOT mark an empty test split as evidence of test isolation (reports NOT_APPLICABLE).
    """
    has_test = len(test) > 0

    train_vids = {str(s.video_path.resolve()) if s.video_path else "" for s in train} - {""}
    val_vids = {str(s.video_path.resolve()) if s.video_path else "" for s in val} - {""}
    test_vids = {str(s.video_path.resolve()) if s.video_path else "" for s in test} - {""}

    train_recs = {(s.subject, s.cam, s.scene, s.session) for s in train}
    val_recs = {(s.subject, s.cam, s.scene, s.session) for s in val}
    test_recs = {(s.subject, s.cam, s.scene, s.session) for s in test}

    train_sessions = {(s.subject, s.scene, s.session) for s in train}
    val_sessions = {(s.subject, s.scene, s.session) for s in val}
    test_sessions = {(s.subject, s.scene, s.session) for s in test}

    results: Dict[str, Any] = {
        "test_split_present": has_test,
        "video_disjoint_train_val": len(train_vids & val_vids) == 0,
        "video_disjoint_train_test": (len(train_vids & test_vids) == 0) if has_test else None,
        "video_disjoint_val_test": (len(val_vids & test_vids) == 0) if has_test else None,
        "recording_disjoint_train_val": len(train_recs & val_recs) == 0,
        "recording_disjoint_train_test": (len(train_recs & test_recs) == 0) if has_test else None,
        "recording_disjoint_val_test": (len(val_recs & test_recs) == 0) if has_test else None,
        "session_disjoint_train_val": len(train_sessions & val_sessions) == 0,
        "session_disjoint_train_test": (len(train_sessions & test_sessions) == 0) if has_test else None,
        "session_disjoint_val_test": (len(val_sessions & test_sessions) == 0) if has_test else None,
    }

    norm_strategy = strategy.lower()
    if norm_strategy in ["subject"]:
        train_subjs = {s.subject for s in train}
        val_subjs = {s.subject for s in val}
        test_subjs = {s.subject for s in test}
        results["subject_disjoint_train_val"] = len(train_subjs & val_subjs) == 0
        results["subject_disjoint_train_test"] = (len(train_subjs & test_subjs) == 0) if has_test else None
        results["subject_disjoint_val_test"] = (len(val_subjs & test_subjs) == 0) if has_test else None

    elif norm_strategy in ["cross_view", "view", "official_x_view", "official_view"]:
        train_cams = {s.cam for s in train}
        val_cams = {s.cam for s in val}
        test_cams = {s.cam for s in test}
        results["view_disjoint_train_val"] = len(train_cams & val_cams) == 0
        results["view_disjoint_train_test"] = (len(train_cams & test_cams) == 0) if has_test else None

    elif norm_strategy in ["scene", "cross_scene"]:
        train_scenes = {s.scene for s in train}
        val_scenes = {s.scene for s in val}
        test_scenes = {s.scene for s in test}
        results["scene_disjoint_train_val"] = len(train_scenes & val_scenes) == 0
        results["scene_disjoint_train_test"] = (len(train_scenes & test_scenes) == 0) if has_test else None

    elif norm_strategy in ["official_x_scene", "official_scene", "official_cross_scene"]:
        # Official MMAct cross-scene protocol enforces recording and session isolation
        # Note: scene3 appears in both train (sessions 1-3) and val (sessions 4-5) in official files
        train_scenes = {s.scene for s in train}
        val_scenes = {s.scene for s in val}
        results["official_scene_overlap_notes"] = f"Train scenes: {train_scenes}, Val scenes: {val_scenes}"

    # Determine required checks based strictly on the selected protocol semantics
    required_checks = [
        "video_disjoint_train_val",
        "recording_disjoint_train_val",
    ]
    if has_test:
        required_checks.extend([
            "video_disjoint_train_test",
            "video_disjoint_val_test",
            "recording_disjoint_train_test",
            "recording_disjoint_val_test",
            "session_disjoint_val_test",
        ])

    if norm_strategy in ["subject"]:
        required_checks.extend(["subject_disjoint_train_val", "session_disjoint_train_val"])
        if has_test:
            required_checks.extend(["subject_disjoint_train_test", "subject_disjoint_val_test", "session_disjoint_train_test"])

    elif norm_strategy in ["cross_view", "view", "official_x_view", "official_view"]:
        required_checks.extend(["view_disjoint_train_val"])
        if has_test:
            required_checks.extend(["view_disjoint_train_test"])

    elif norm_strategy in ["scene", "cross_scene"]:
        required_checks.extend(["scene_disjoint_train_val", "session_disjoint_train_val"])
        if has_test:
            required_checks.extend(["scene_disjoint_train_test", "session_disjoint_train_test"])

    elif norm_strategy in ["official_x_scene", "official_scene", "official_cross_scene"]:
        required_checks.extend(["session_disjoint_train_val"])
        if has_test:
            required_checks.extend(["session_disjoint_train_test"])

    elif norm_strategy == "session":
        required_checks.extend(["session_disjoint_train_val"])
        if has_test:
            required_checks.extend(["session_disjoint_train_test"])

    # Overall isolation validity check: all required protocol checks must evaluate to True
    results["all_isolation_checks_passed"] = all(
        results.get(k) is True for k in required_checks
    )
    results["required_isolation_checks"] = required_checks

    return results


def verify_split_representation(
    train: List[PairedSample],
    val: List[PairedSample],
    test: List[PairedSample],
    expected_classes: Optional[List[str]] = None,
    required_sensor_modalities: Optional[List[str]] = None,
    check_disk_files: bool = True,
    enforce: bool = False,
) -> Dict[str, Any]:
    """
    Audits class representation, sample availability, and individual modality integrity across splits.
    Checks:
    - Non-empty train and validation splits
    - Coverage and per-class counts for all target classes
    - Detection of unknown labels and missing/unassigned labels
    - Individual validation of each required sensor modality ('acc_phone', 'acc_watch', 'gyro')
    - Physical existence checks for video and sensor files on disk
    - Clear distinction between missing modality, missing file, and class absence
    """
    if expected_classes is None:
        expected_classes = VERIFIED_TARGET_CLASSES
    if required_sensor_modalities is None:
        required_sensor_modalities = ["acc_phone", "acc_watch", "gyro"]

    exp_set = set(expected_classes)
    has_test = len(test) > 0

    # 1. Class distributions and label validation
    train_dist: Dict[str, int] = {c: 0 for c in expected_classes}
    val_dist: Dict[str, int] = {c: 0 for c in expected_classes}
    test_dist: Dict[str, int] = {c: 0 for c in expected_classes} if has_test else {}

    unknown_labels: List[Tuple[str, str, str]] = []  # (split, sample_id, label)
    missing_labels: List[Tuple[str, str]] = []  # (split, sample_id)

    for sp_name, s_list, d_dict in [("train", train, train_dist), ("val", val, val_dist), ("test", test, test_dist)]:
        for s in s_list:
            if not s.action or s.label_idx < 0:
                missing_labels.append((sp_name, s.sample_id))
            elif s.action not in exp_set:
                unknown_labels.append((sp_name, s.sample_id, s.action))
            else:
                d_dict[s.action] = d_dict.get(s.action, 0) + 1

    train_classes = {s.action for s in train if s.action in exp_set}
    val_classes = {s.action for s in val if s.action in exp_set}
    test_classes = {s.action for s in test if s.action in exp_set} if has_test else set()

    missing_train = sorted(list(exp_set - train_classes))
    missing_val = sorted(list(exp_set - val_classes))
    missing_test = sorted(list(exp_set - test_classes)) if has_test else []

    # 2. Modality and file validation
    missing_modalities: List[Tuple[str, str, str]] = []  # (split, sample_id, modality)
    missing_video_files: List[Tuple[str, str, str]] = []  # (split, sample_id, path)
    missing_sensor_files: List[Tuple[str, str, str, str]] = []  # (split, sample_id, modality, path)

    for sp_name, s_list in [("train", train), ("val", val), ("test", test)]:
        for s in s_list:
            # Video check
            if not s.video_path:
                missing_video_files.append((sp_name, s.sample_id, "None"))
            elif check_disk_files and not Path(s.video_path).exists():
                missing_video_files.append((sp_name, s.sample_id, str(s.video_path)))

            # Individual sensor modality checks
            for mod in required_sensor_modalities:
                if mod not in s.sensor_paths or not s.sensor_paths[mod]:
                    missing_modalities.append((sp_name, s.sample_id, mod))
                elif check_disk_files:
                    f_path = Path(s.sensor_paths[mod])
                    if not f_path.exists():
                        missing_sensor_files.append((sp_name, s.sample_id, mod, str(f_path)))

    all_modalities_present = (len(missing_modalities) == 0 and len(missing_sensor_files) == 0 and len(missing_video_files) == 0)

    report = {
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "test_split_present": has_test,
        "all_classes_present_train": len(missing_train) == 0,
        "all_classes_present_val": len(missing_val) == 0,
        "all_classes_present_test": (len(missing_test) == 0) if has_test else None,
        "missing_classes_train": missing_train,
        "missing_classes_val": missing_val,
        "missing_classes_test": missing_test,
        "train_class_distribution": train_dist,
        "val_class_distribution": val_dist,
        "test_class_distribution": test_dist,
        "unknown_labels": unknown_labels,
        "missing_labels": missing_labels,
        "missing_modality_incidents": missing_modalities,
        "missing_video_files": missing_video_files,
        "missing_sensor_files": missing_sensor_files,
        "all_modalities_verified": all_modalities_present,
        "modalities_paired_train": not any(item[0] == "train" for item in missing_modalities),
        "modalities_paired_val": not any(item[0] == "val" for item in missing_modalities),
        "modalities_paired_test": (not any(item[0] == "test" for item in missing_modalities)) if has_test else True,
    }

    if enforce:
        errors = []
        if len(train) == 0:
            errors.append("Train split is empty!")
        if len(val) == 0:
            errors.append("Validation split is empty!")
        if missing_train:
            errors.append(f"Target classes missing from train split: {missing_train}")
        if missing_val:
            errors.append(f"Target classes missing from validation split: {missing_val}")
        if unknown_labels:
            errors.append(f"Unknown labels found: {len(unknown_labels)} samples (e.g. {unknown_labels[:3]})")
        if missing_labels:
            errors.append(f"Missing labels found: {len(missing_labels)} samples")
        if missing_modalities:
            errors.append(f"Missing required sensor modalities: {len(missing_modalities)} incidents (e.g. {missing_modalities[:3]})")
        if missing_video_files:
            errors.append(f"Missing video files on disk: {len(missing_video_files)} files (e.g. {missing_video_files[:3]})")
        if missing_sensor_files:
            errors.append(f"Missing sensor files on disk: {len(missing_sensor_files)} files (e.g. {missing_sensor_files[:3]})")

        if errors:
            raise ValueError("Split representation verification failed:\n  - " + "\n  - ".join(errors))

    return report
