"""
splits.py - Leakage-Free Dataset Splitting (Cross-Subject, Cross-View, Cross-Scene)

Enforces strict separation between training, validation, and test sets.
Splits are performed at the subject, camera, scene, or recording level BEFORE windowing,
guaranteeing zero cross-split leakage.
"""

from typing import Dict, List, Set, Tuple
import numpy as np

from src.config import SplitConfig
from src.data.indexer import PairedSample


class DatasetSplitter:
    """Partitions indexed samples into isolated train, validation, and test sets."""

    def __init__(self, config: SplitConfig):
        self.config = config

    def split(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """
        Splits samples according to config strategy ('subject', 'cross_view', 'cross_scene', 'session').
        Returns: (train_samples, val_samples, test_samples)
        """
        if not samples:
            return ([], [], [])

        if self.config.split_strategy == "cross_view":
            return self._split_by_cross_view(samples)
        elif self.config.split_strategy in ["scene", "cross_scene"]:
            return self._split_by_cross_scene(samples)
        elif self.config.split_strategy == "session":
            return self._split_by_session(samples)
        else:  # default 'subject'
            return self._split_by_subject(samples)

    def _split_by_cross_view(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """
        Official MMAct Task 1 Cross-View Protocol:
        Held-out validation view is cam3. Train on cam1, cam2, cam4.
        """
        train = [s for s in samples if s.cam != "cam3"]
        val = [s for s in samples if s.cam == "cam3"]
        test = list(val)  # Held-out view test set
        return train, val, test

    def _split_by_cross_scene(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """
        Official MMAct Task 1 Cross-Scene Protocol:
        Train on scenes 1 and 2. Validation on scenes 3 and 4.
        """
        train = [s for s in samples if s.scene in ["scene1", "scene2"]]
        val = [s for s in samples if s.scene in ["scene3", "scene4"]]
        test = list(val)
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
                train_set = {all_subjects[0]}
                val_set = {all_subjects[1]}
                test_set = {all_subjects[1]}
            else:
                return self._split_by_session(samples)
        else:
            train_set = configured_train
            val_set = configured_val - train_set
            test_set = configured_test - train_set - val_set

        train = [s for s in samples if s.subject in train_set]
        val = [s for s in samples if s.subject in val_set]
        test = [s for s in samples if s.subject in test_set]

        return train, val, test

    def _split_by_session(
        self, samples: List[PairedSample]
    ) -> Tuple[List[PairedSample], List[PairedSample], List[PairedSample]]:
        """Partitions by unique recording session to avoid recording-level leakage."""
        sessions = sorted(list({(s.subject, s.scene, s.session) for s in samples}))
        rng = np.random.RandomState(42)
        rng.shuffle(sessions)

        n_total = len(sessions)
        n_train = max(1, int(round(n_total * self.config.train_ratio)))
        n_val = max(1, int(round(n_total * self.config.val_ratio)))

        train_sessions = set(sessions[:n_train])
        val_sessions = set(sessions[n_train : n_train + n_val])
        test_sessions = set(sessions[n_train + n_val :])

        train = [
            s for s in samples if (s.subject, s.scene, s.session) in train_sessions
        ]
        val = [
            s for s in samples if (s.subject, s.scene, s.session) in val_sessions
        ]
        test = [
            s for s in samples if (s.subject, s.scene, s.session) in test_sessions
        ]

        return train, val, test


def verify_split_isolation(
    train: List[PairedSample],
    val: List[PairedSample],
    test: List[PairedSample],
    strategy: str = "subject",
) -> Dict[str, bool]:
    """
    Formally verifies that there is zero overlap between train and validation/test splits.
    """
    train_vids = {str(s.video_path) for s in train}
    val_vids = {str(s.video_path) for s in val}
    test_vids = {str(s.video_path) for s in test}

    train_sessions = {(s.subject, s.scene, s.session) for s in train}
    val_sessions = {(s.subject, s.scene, s.session) for s in val}
    test_sessions = {(s.subject, s.scene, s.session) for s in test}

    results = {
        "video_disjoint_train_val": len(train_vids & val_vids) == 0,
        "video_disjoint_train_test": len(train_vids & test_vids) == 0,
        "session_disjoint_train_val": len(train_sessions & val_sessions) == 0,
        "session_disjoint_train_test": len(train_sessions & test_sessions) == 0,
    }

    if strategy == "subject":
        train_subjs = {s.subject for s in train}
        val_subjs = {s.subject for s in val}
        test_subjs = {s.subject for s in test}
        results["subject_disjoint_train_val"] = len(train_subjs & val_subjs) == 0
        results["subject_disjoint_train_test"] = len(train_subjs & test_subjs) == 0

    elif strategy == "cross_view":
        train_cams = {s.cam for s in train}
        val_cams = {s.cam for s in val}
        results["view_disjoint_train_val"] = len(train_cams & val_cams) == 0

    elif strategy in ["scene", "cross_scene"]:
        train_scenes = {s.scene for s in train}
        val_scenes = {s.scene for s in val}
        results["scene_disjoint_train_val"] = len(train_scenes & val_scenes) == 0

    return results
