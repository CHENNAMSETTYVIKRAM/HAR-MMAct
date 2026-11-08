"""
config.py - Central Configuration and Reproducibility Controls for HAR_MMAct

All paths, random seeds, hyperparameters, target class mappings, and modality
specifications are centrally declared here.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 10 Strictly Verified Target Classes from docs/LABEL_MAPPING.md
VERIFIED_TARGET_CLASSES: List[str] = [
    "crouching",
    "jumping",
    "kicking",
    "pointing",
    "running",
    "sitting",
    "standing",
    "using_phone",
    "walking",
    "waving_hand",
]

TARGET_CLASS_TO_IDX: Dict[str, int] = {
    cls_name: idx for idx, cls_name in enumerate(VERIFIED_TARGET_CLASSES)
}
IDX_TO_TARGET_CLASS: Dict[int, str] = {
    idx: cls_name for cls_name, idx in TARGET_CLASS_TO_IDX.items()
}

# Sensor modality column specifications based on official challenge toolkit
SENSOR_COLUMNS = {
    "acc_phone": ["time", "x", "y", "z"],
    "acc_watch": ["time", "x", "y", "z"],
    "gyro": ["time", "x", "y", "z"],
    "orientation": ["time", "azimuth", "pitch", "roll"],
}

SENSOR_RATES_HZ = {
    "acc_phone": 100,
    "acc_watch": 100,
    "gyro": 50,
    "orientation": 50,
}


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic CUDA operations if available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class VideoConfig:
    num_frames: int = 16
    frame_height: int = 112
    frame_width: int = 112
    channels: int = 3
    sampling_strategy: str = (
        "uniform"  # 'uniform', 'consecutive', or 'random_window'
    )
    normalize_mean: List[float] = field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    normalize_std: List[float] = field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )


@dataclass
class SensorConfig:
    # Modalites to include in paired sensor stream
    modalities: List[str] = field(
        default_factory=lambda: ["acc_phone", "acc_watch", "gyro"]
    )
    primary_rate_hz: int = 100
    window_duration_sec: float = 1.0  # 1.0 sec window
    window_size: int = 100  # 100 samples at 100 Hz
    step_size: int = 25  # 75% overlap (stride 25 samples)
    convert_acc_to_g: bool = (
        True  # MMAct accelerometer divide by 9.8 (m/s^2 -> g)
    )
    fill_strategy: str = "bfill"  # backward fill gap alignment ('bfill' or 'bfill_ffill_bounded')
    max_ffill_limit: Optional[int] = (
        None  # None -> auto-calculated (e.g. 2 for 50 Hz -> 100 Hz)
    )


@dataclass
class SplitConfig:
    # Split strategies: 'subject', 'cross_view', 'cross_scene', 'session', 'official_x_scene', 'official_x_view'
    split_strategy: str = "subject"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    partition_val_for_test: bool = False
    split_dir: Optional[Path] = None
    dataset_root: Optional[Path] = None
    seed: int = 42
    # Explicit subject partition defaults (1-20 total subjects in MMAct)
    train_subjects: List[str] = field(
        default_factory=lambda: [f"subject{i}" for i in range(1, 15)]
    )
    val_subjects: List[str] = field(
        default_factory=lambda: [f"subject{i}" for i in range(15, 18)]
    )
    test_subjects: List[str] = field(
        default_factory=lambda: [f"subject{i}" for i in range(18, 21)]
    )


@dataclass
class ProjectConfig:
    project_root: Path = PROJECT_ROOT
    dataset_root: Path = PROJECT_ROOT / "data" / "genuine_mmact"
    synthetic_root: Path = PROJECT_ROOT / "data" / "synthetic_smoke_test"
    cache_root: Path = PROJECT_ROOT / "outputs" / "cache"
    output_root: Path = PROJECT_ROOT / "outputs"

    random_seed: int = 42
    video: VideoConfig = field(default_factory=VideoConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    batch_size: int = 8
    num_workers: int = 0  # 0 is most robust on Windows
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Create output directories if needed
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
