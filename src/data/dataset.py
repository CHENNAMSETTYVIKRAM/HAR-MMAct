"""
dataset.py - PyTorch Multimodal Dataset with Preprocessing Cache for HAR_MMAct
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from src.data.indexer import PairedSample
from src.data.video_loader import VideoLoader
from src.data.sensor_loader import SensorLoader


class MMActMultimodalDataset(Dataset):
    """
    PyTorch Dataset providing paired (video, sensor) data tensors and class labels.
    Supports caching preprocessed tensors on disk to avoid repeatedly decoding videos.
    """

    def __init__(
        self,
        samples: List[PairedSample],
        video_loader: Optional[VideoLoader] = None,
        sensor_loader: Optional[SensorLoader] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cache: bool = True,
    ):
        self.samples = samples
        self.video_loader = video_loader or VideoLoader()
        self.sensor_loader = sensor_loader or SensorLoader()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_cache = enable_cache and (self.cache_dir is not None)

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cache_path(self, sample: PairedSample) -> Optional[Path]:
        if not self.enable_cache or self.cache_dir is None:
            return None
        # Hash sample ID to avoid long filenames or invalid characters
        sample_hash = hashlib.md5(sample.sample_id.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{sample_hash}.pt"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Returns:
            video_tensor  : (C, T, H, W) float32 tensor
            sensor_tensor : (num_sensor_channels, window_size) float32 tensor
            label_idx     : int target class index in [0, 9]
            sample_id     : str sample identifier
        """
        sample = self.samples[idx]
        cache_path = self._get_cache_path(sample)

        if cache_path and cache_path.exists():
            try:
                cached = torch.load(cache_path)
                return (
                    cached["video"],
                    cached["sensor"],
                    sample.label_idx,
                    sample.sample_id,
                )
            except Exception:
                # If cached tensor is corrupted, recompute
                pass

        # Load video frames
        video_tensor = self.video_loader.load_and_preprocess(sample.video_path)

        # Load and align sensor windows
        sensor_tensor = self.sensor_loader.load_and_preprocess(
            sample.sensor_paths
        )

        # Save to cache if enabled
        if cache_path:
            try:
                torch.save(
                    {"video": video_tensor, "sensor": sensor_tensor},
                    cache_path,
                )
            except Exception:
                pass

        return video_tensor, sensor_tensor, sample.label_idx, sample.sample_id
