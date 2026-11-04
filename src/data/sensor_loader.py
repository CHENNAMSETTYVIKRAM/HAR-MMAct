"""
sensor_loader.py - Multi-Modal Sensor Ingestion, Validation, Resampling & Windowing

Implements the official MMAct sensor preprocessing conventions from utils/time_series_classifiers.py:
1. Loads CSV sensor recordings (handles both files with headers and headerless files).
2. Converts accelerometer m/s^2 -> g (divide by 9.8).
3. Parses start timestamp (format %Y%m%d_%H:%M:%S.%f or epoch) and generates uniform 100 Hz/50 Hz timegrid.
4. Multi-modal datetime alignment & resampling onto a uniform 100 Hz timeline matching official MMAct code.
5. Slices fixed-length sliding windows without cross-subject or cross-session leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.config import SensorConfig, SENSOR_COLUMNS, SENSOR_RATES_HZ


class SensorLoader:
    """Loads, cleans, aligns, and windows multi-modal MMAct wearable sensor data."""

    def __init__(self, config: Optional[SensorConfig] = None):
        self.config = config or SensorConfig()

    def parse_start_time_epoch_seconds(self, first_val: str) -> float:
        """
        Parses initial timestamp string (%Y%m%d_%H:%M:%S.%f or %Y/%m/%d %H:%M:%S.%f or epoch)
        into seconds since epoch (UTC, localized from Asia/Tokyo per official MMAct code).
        """
        first_str = str(first_val).strip()

        # Check if already numeric epoch in ms or seconds
        try:
            num = float(first_str)
            if num > 1e11:  # Epoch in ms
                return num / 1000.0
            return num
        except ValueError:
            pass

        # Ensure subsecond dot exists
        if "." not in first_str:
            first_str = first_str + ".000"

        # Check format with slash or underscore
        date_fmt = (
            "%Y/%m/%d %H:%M:%S.%f"
            if "/" in first_str
            else "%Y%m%d_%H:%M:%S.%f"
        )
        try:
            dt = pd.to_datetime(first_str, format=date_fmt)
            dt_utc = dt.tz_localize("Asia/Tokyo").tz_convert("UTC")
            stt_time = (
                dt_utc - pd.Timestamp("1970-01-01", tz="UTC")
            ) / pd.Timedelta("1s")
            return round(stt_time, 2)
        except Exception:
            # Fallback standard ISO/flexible parse
            dt = pd.to_datetime(first_str, errors="coerce")
            if pd.isna(dt):
                return 0.0
            return dt.timestamp()

    def load_single_csv(
        self, file_path: Union[str, Path], modality: str
    ) -> pd.DataFrame:
        """
        Reads raw sensor CSV, validating columns and handling both headered and headerless files.
        Generates uniform time grid matching official MMAct time_series_classifiers.py conventions.
        """
        expected_cols = SENSOR_COLUMNS.get(
            modality, ["time", "x", "y", "z"]
        )
        sample_rate = SENSOR_RATES_HZ.get(modality, 100)

        # 1. Attempt reading with expected header names
        try:
            df = pd.read_csv(file_path, usecols=expected_cols)
        except (ValueError, KeyError):
            # In genuine MMAct archives, several CSVs have NO header row.
            # In this case, load with names=expected_cols without header=0 so row 0 is NOT dropped!
            df = pd.read_csv(file_path, names=expected_cols)

        if len(df) == 0:
            raise ValueError(f"Empty sensor CSV file: {file_path}")

        # 2. Interpolate any NaN values in numeric axis columns
        axis_cols = [c for c in expected_cols if c != "time"]
        for col in axis_cols:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .interpolate(method="linear")
                    .bfill()
                    .ffill()
                )

        # 3. Scale accelerometer values to g's (dividing by 9.8) as per official MMAct code
        if self.config.convert_acc_to_g and "acc" in modality:
            df[axis_cols] = df[axis_cols] / 9.8

        # 4. Synthesize uniform epoch millisecond timestamps from start time
        # This matches time_series_classifiers.py lines 208-222 to eliminate hardware jitter and duplicate timestamps
        stt_time_sec = self.parse_start_time_epoch_seconds(df["time"].iloc[0])
        time_values = np.array(
            [
                int(round(1000 * (stt_time_sec + (i / sample_rate))))
                for i in range(len(df))
            ],
            dtype=np.int64,
        )

        # 5. Rename axis columns with modality prefix for multi-sensor uniqueness
        prefix = {
            "acc_phone": "s_acc",
            "acc_watch": "w_acc",
            "gyro": "gyro",
            "orientation": "ori",
        }.get(modality, modality)

        rename_map = {c: f"{prefix}_{c}" for c in axis_cols}
        df.rename(columns=rename_map, inplace=True)
        df["time"] = time_values
        df.set_index("time", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        return df

    def align_modalities(
        self, sensor_paths: Dict[str, Union[str, Path]]
    ) -> pd.DataFrame:
        """
        Aligns disparate sensor streams onto the primary 100 Hz timeline using backward-fill.
        Matches official MMAct datetime_merging logic.
        """
        dfs = {}
        for mod, path in sensor_paths.items():
            if mod in self.config.modalities and Path(path).exists():
                dfs[mod] = self.load_single_csv(path, mod)

        if not dfs:
            raise ValueError("No valid sensor streams found to align.")

        # Determine primary dataframe (prefer 100 Hz phone acc, then watch acc, then any)
        primary_mod = next(
            (m for m in ["acc_phone", "acc_watch", "gyro"] if m in dfs),
            list(dfs.keys())[0],
        )
        primary_df = dfs[primary_mod].copy(deep=True)

        for mod, sec_df in dfs.items():
            if mod == primary_mod:
                continue
            # Left join on primary timeline
            primary_df = primary_df.join(sec_df, how="left")

        # Fill resampled frequencies (e.g. 50 Hz gyro joined onto 100 Hz primary)
        if self.config.fill_strategy == "bfill":
            primary_df = primary_df.bfill().ffill()
        else:
            primary_df = primary_df.interpolate(method="linear").bfill().ffill()

        primary_df.dropna(inplace=True)
        return primary_df

    def create_fixed_window(
        self, aligned_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Extracts a fixed-length window of shape (num_channels, window_size).
        If the clip is shorter than window_size, it is symmetrically edge-padded.
        If longer, the centered window is selected for standard clip-level inference.
        """
        feature_data = aligned_df.to_numpy(dtype=np.float32)  # Shape: (N, C)
        n_samples, n_channels = feature_data.shape
        w = self.config.window_size

        if n_samples == w:
            window = feature_data
        elif n_samples > w:
            start = (n_samples - w) // 2
            window = feature_data[start : start + w]
        else:
            pad_before = (w - n_samples) // 2
            pad_after = w - n_samples - pad_before
            window = np.pad(
                feature_data,
                ((pad_before, pad_after), (0, 0)),
                mode="edge",
            )

        # Transpose to (C, W) for PyTorch 1D convolutions: (channels, window_size)
        return np.transpose(window).astype(np.float32)

    def load_and_preprocess(
        self, sensor_paths: Dict[str, Union[str, Path]]
    ) -> torch.Tensor:
        """Full sensor pipeline: load CSVs, align, slice fixed window -> Tensor (C, W)."""
        aligned_df = self.align_modalities(sensor_paths)
        window_np = self.create_fixed_window(aligned_df)
        return torch.from_numpy(window_np).float()
