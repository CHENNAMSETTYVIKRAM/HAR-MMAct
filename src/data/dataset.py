"""
dataset.py - PyTorch Multimodal Dataset with Atomic Preprocessing Cache for HAR_MMAct

Features:
- Deterministic, versioned, and configuration-aware cache keys (sample_id, video params, sensor params).
- Atomic disk caching (temporary file write + os.replace) to prevent partial/corrupted entries.
- Integrity verification on cache load (shape checking, NaN detection, corrupted file removal).
- Storage safety check (verifies sufficient free space via shutil.disk_usage before caching).
- Reusable across all model experiments (Custom CNN, Tuned CNN, MobileNetV2, InceptionTime, Multimodal).
- Cache hit/miss/write/corruption metrics tracking.
- Pre-caching warm-up utility.
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from src.data.indexer import PairedSample
from src.data.video_loader import VideoLoader
from src.data.sensor_loader import SensorLoader

CACHE_SCHEMA_VERSION = "cache_v2"
MIN_FREE_DISK_BYTES = 250 * 1024 * 1024  # 250 MB safety margin


class MMActMultimodalDataset(Dataset):
    """
    PyTorch Dataset providing paired (video, sensor) data tensors and class labels.
    Supports persistent, atomic, versioned caching on disk to avoid repeated decoding.
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

        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_writes = 0
        self.cache_corruptions = 0

        # Precompute config hash prefix for invalidation safety
        vc = self.video_loader.config
        sc = self.sensor_loader.config
        vh = getattr(vc, "frame_height", 112)
        vw = getattr(vc, "frame_width", 112)
        sr = getattr(sc, "primary_rate_hz", 100)
        mods = "-".join(sorted(getattr(sc, "modalities", [])))
        config_descriptor = (
            f"{CACHE_SCHEMA_VERSION}|"
            f"vf={vc.num_frames}|vh={vh}|vw={vw}|vs={vc.sampling_strategy}|"
            f"sw={sc.window_size}|sr={sr}|sm={mods}"
        )
        self.config_hash = hashlib.sha256(config_descriptor.encode("utf-8")).hexdigest()[:12]

        if self.enable_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cache_path(self, sample: PairedSample) -> Optional[Path]:
        if not self.enable_cache or self.cache_dir is None:
            return None

        # Build stable cache identifier incorporating sample identity + config hash
        source_id = sample.sample_id
        # Include video mtime or file size if available for source-identity safety
        v_meta = ""
        try:
            vp = Path(sample.video_path)
            if vp.exists():
                v_meta = f"{vp.stat().st_size}_{int(vp.stat().st_mtime)}"
        except Exception:
            pass

        key_str = f"{self.config_hash}:{source_id}:{v_meta}"
        sample_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{sample_hash}.pt"

    def _validate_cached_entry(self, cached: Dict[str, torch.Tensor]) -> bool:
        """Validates that loaded cached tensor dictionary is complete and uncorrupted."""
        if not isinstance(cached, dict):
            return False
        if "video" not in cached or "sensor" not in cached:
            return False

        v = cached["video"]
        s = cached["sensor"]
        if not isinstance(v, torch.Tensor) or not isinstance(s, torch.Tensor):
            return False

        vc = self.video_loader.config
        sc = self.sensor_loader.config

        expected_v_channels = getattr(vc, "channels", 3)
        expected_v_frames = getattr(vc, "num_frames", 16)
        expected_v_h = getattr(vc, "frame_height", 112)
        expected_v_w = getattr(vc, "frame_width", 112)

        expected_s_channels = getattr(sc, "num_channels", len(getattr(sc, "modalities", [])) * 3)
        if expected_s_channels == 0:
            expected_s_channels = 9
        expected_s_window = getattr(sc, "window_size", 100)

        # Check expected shape: v is (C, T, H, W)
        if v.ndim != 4 or v.shape[0] != expected_v_channels or v.shape[1] != expected_v_frames:
            return False
        if v.shape[2] != expected_v_h or v.shape[3] != expected_v_w:
            return False

        # s is (C, W)
        if s.ndim != 2 or s.shape[0] != expected_s_channels or s.shape[1] != expected_s_window:
            return False

        # Check for NaN / Inf corruption
        if torch.isnan(v).any() or torch.isinf(v).any():
            return False
        if torch.isnan(s).any() or torch.isinf(s).any():
            return False

        return True

    def _has_sufficient_disk_space(self) -> bool:
        """Checks if cache directory has sufficient free disk space."""
        if not self.cache_dir:
            return False
        try:
            free = shutil.disk_usage(str(self.cache_dir)).free
            return free >= MIN_FREE_DISK_BYTES
        except Exception:
            return True

    def get_cache_stats(self) -> Dict[str, int]:
        """Returns current cache operational metrics."""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "writes": self.cache_writes,
            "corruptions": self.cache_corruptions,
            "total_queries": self.cache_hits + self.cache_misses,
        }

    def warm_up_cache(self, verbose: bool = True) -> Dict[str, int]:
        """
        Sequentially pre-loads and caches all samples.
        Useful for warm-up before multi-epoch training to avoid I/O bottlenecks.
        """
        if not self.enable_cache:
            return self.get_cache_stats()

        count = len(self.samples)
        if verbose:
            print(f"[Cache Warm-Up] Pre-caching {count} samples at {self.cache_dir}...")

        for idx in range(count):
            _ = self.__getitem__(idx)
            if verbose and (idx + 1) % 50 == 0:
                stats = self.get_cache_stats()
                print(f"  Processed {idx + 1}/{count} samples (Hits: {stats['hits']}, Misses: {stats['misses']})")

        stats = self.get_cache_stats()
        if verbose:
            print(f"[Cache Warm-Up Complete] Total: {count} | Hits: {stats['hits']} | Misses/Writes: {stats['writes']}")
        return stats

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

        # 1. Attempt to load from valid cache
        if cache_path and cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu")
                if self._validate_cached_entry(cached):
                    self.cache_hits += 1
                    return (
                        cached["video"],
                        cached["sensor"],
                        sample.label_idx,
                        sample.sample_id,
                    )
                else:
                    # Invalid or stale cache entry detected -> remove and recompute
                    self.cache_corruptions += 1
                    cache_path.unlink(missing_ok=True)
            except Exception:
                self.cache_corruptions += 1
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    pass

        self.cache_misses += 1

        # 2. Recompute video and sensor tensors
        video_tensor = self.video_loader.load_and_preprocess(sample.video_path)
        sensor_tensor = self.sensor_loader.load_and_preprocess(sample.sensor_paths)

        # 3. Save to cache atomically if disk space permits
        if cache_path and self._has_sufficient_disk_space():
            pid = os.getpid()
            temp_cache_path = cache_path.with_suffix(f".tmp_{pid}_{os.urandom(4).hex()}")
            try:
                payload = {"video": video_tensor, "sensor": sensor_tensor}
                torch.save(payload, temp_cache_path)
                os.replace(temp_cache_path, cache_path)
                self.cache_writes += 1
            except Exception:
                if temp_cache_path.exists():
                    try:
                        temp_cache_path.unlink()
                    except Exception:
                        pass

        return video_tensor, sensor_tensor, sample.label_idx, sample.sample_id
