"""
infer.py - Standalone Inference CLI for Multimodal, Video-Only, and Sensor-Only HAR
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import VERIFIED_TARGET_CLASSES, VideoConfig, SensorConfig
from src.data.video_loader import VideoLoader
from src.data.sensor_loader import SensorLoader
from src.models.fusion_model import MultimodalHARModel


def parse_args():
    parser = argparse.ArgumentParser(description="Predict activity from video and/or sensor recordings")
    parser.add_argument("--video", type=str, default=None, help="Path to input video (.mp4)")
    parser.add_argument("--sensor_phone", type=str, default=None, help="Path to acc_phone CSV")
    parser.add_argument("--sensor_watch", type=str, default=None, help="Path to acc_watch CSV")
    parser.add_argument("--sensor_gyro", type=str, default=None, help="Path to gyro CSV")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Checkpoint path")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    has_video = args.video is not None and Path(args.video).exists()
    sensor_paths = {}
    if args.sensor_phone and Path(args.sensor_phone).exists():
        sensor_paths["acc_phone"] = Path(args.sensor_phone)
    if args.sensor_watch and Path(args.sensor_watch).exists():
        sensor_paths["acc_watch"] = Path(args.sensor_watch)
    if args.sensor_gyro and Path(args.sensor_gyro).exists():
        sensor_paths["gyro"] = Path(args.sensor_gyro)

    has_sensors = len(sensor_paths) > 0

    if not has_video and not has_sensors:
        print("[ERROR] Please provide either --video, or at least one sensor file (--sensor_phone, --sensor_watch, --sensor_gyro).")
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_cfg = checkpoint.get("config", {})
    classes = checkpoint.get("classes", VERIFIED_TARGET_CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using inference device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using inference device: {device} (CPU)")
    model = MultimodalHARModel(
        num_classes=len(classes),
        video_channels=3,
        sensor_channels=9,
        video_embed_dim=128,
        sensor_embed_dim=128,
        fusion_hidden_dim=128,
        video_backbone=saved_cfg.get("video_backbone", "resnet18"),
        sensor_backbone=saved_cfg.get("sensor_backbone", "inception_time"),
        pretrained_video=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    video_tensor = None
    sensor_tensor = None

    if has_video:
        v_loader = VideoLoader(VideoConfig())
        video_tensor = v_loader.load_and_preprocess(args.video).unsqueeze(0).to(device)

    if has_sensors:
        s_loader = SensorLoader(SensorConfig())
        sensor_tensor = s_loader.load_and_preprocess(sensor_paths).unsqueeze(0).to(device)

    print("\n" + "=" * 60)
    print(" HAR_MMAct: Inference Prediction")
    print("=" * 60)

    if has_video and has_sensors:
        print("[Mode: Multimodal Fused Inference]")
        probs = model.predict_fused(video_tensor, sensor_tensor)[0].cpu().numpy()
    elif has_video:
        print("[Mode: Video-Only Inference]")
        probs = model.predict_video_only(video_tensor)[0].cpu().numpy()
    else:
        print("[Mode: Sensor-Only Inference]")
        probs = model.predict_sensor_only(sensor_tensor)[0].cpu().numpy()

    top_indices = np.argsort(probs)[::-1][: args.top_k]
    print(f"\nTop-{args.top_k} Activity Predictions:")
    for rank, idx in enumerate(top_indices, start=1):
        label = classes[idx]
        conf = probs[idx] * 100.0
        print(f"  {rank}. {label:<15} : {conf:6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
