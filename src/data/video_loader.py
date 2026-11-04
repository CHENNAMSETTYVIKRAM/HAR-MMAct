"""
video_loader.py - Frame Extraction, Sampling, and Preprocessing for MMAct Videos
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from src.config import VideoConfig


class VideoLoader:
    """Loads video files, uniformly samples fixed frames, resizes, and normalizes."""

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig()
        self.mean = np.array(self.config.normalize_mean, dtype=np.float32)
        self.std = np.array(self.config.normalize_std, dtype=np.float32)

    def load_frames(self, video_path: Union[str, Path]) -> np.ndarray:
        """
        Reads raw video frames from path.
        Returns: np.ndarray of shape (num_frames_target, H, W, 3) in RGB format.
        Features robust fallback for unindexed/corrupted containers where CAP_PROP_FRAME_COUNT is 0 or invalid.
        """
        path_str = str(video_path)
        cap = cv2.VideoCapture(path_str)

        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {path_str}")

        t = self.config.num_frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if frame count property is reliable (> 0)
        if total_frames > 0:
            if total_frames >= t:
                indices = np.linspace(0, total_frames - 1, t, dtype=int)
            else:
                indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
                pad_count = t - total_frames
                indices = np.concatenate(
                    [indices, np.full(pad_count, total_frames - 1, dtype=int)]
                )

            target_set = set(indices)
            frame_dict = {}
            current_idx = 0

            success, frame = cap.read()
            while success:
                if current_idx in target_set:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(
                        frame_rgb,
                        (self.config.frame_width, self.config.frame_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    frame_dict[current_idx] = frame_resized
                current_idx += 1
                if current_idx > max(indices, default=0):
                    break
                success, frame = cap.read()
            cap.release()

            if frame_dict:
                sampled_frames = []
                last_valid = next(iter(frame_dict.values()))
                for idx in indices:
                    if idx in frame_dict:
                        last_valid = frame_dict[idx]
                    sampled_frames.append(last_valid)
                return np.stack(sampled_frames, axis=0)

        # Fallback: sequential frame decoding if CAP_PROP_FRAME_COUNT was <= 0 or index seeking failed
        cap = cv2.VideoCapture(path_str)
        all_frames = []
        success, frame = cap.read()
        while success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(
                frame_rgb,
                (self.config.frame_width, self.config.frame_height),
                interpolation=cv2.INTER_LINEAR,
            )
            all_frames.append(frame_resized)
            success, frame = cap.read()
        cap.release()

        n_decoded = len(all_frames)
        if n_decoded == 0:
            # Absolute fallback if video has 0 decodable frames
            zeros = np.zeros(
                (self.config.frame_height, self.config.frame_width, self.config.channels),
                dtype=np.uint8,
            )
            return np.stack([zeros] * t, axis=0)

        if n_decoded >= t:
            idxs = np.linspace(0, n_decoded - 1, t, dtype=int)
        else:
            idxs = np.linspace(0, n_decoded - 1, n_decoded, dtype=int)
            pad = t - n_decoded
            idxs = np.concatenate([idxs, np.full(pad, n_decoded - 1, dtype=int)])

        sampled = [all_frames[i] for i in idxs]
        return np.stack(sampled, axis=0)  # Shape: (T, H, W, C)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        Normalizes and formats numpy frames into PyTorch tensor.
        Input: frames of shape (T, H, W, C) in range [0, 255]
        Output: torch.Tensor of shape (C, T, H, W) normalized
        """
        frames_float = frames.astype(np.float32) / 255.0
        normalized = (frames_float - self.mean) / self.std
        transposed = np.transpose(normalized, (3, 0, 1, 2))
        return torch.from_numpy(transposed).float()

    def load_and_preprocess(self, video_path: Union[str, Path]) -> torch.Tensor:
        """Full pipeline: load, sample, resize, normalize -> Tensor (C, T, H, W)."""
        frames = self.load_frames(video_path)
        return self.preprocess(frames)
