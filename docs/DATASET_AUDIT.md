# MMAct Dataset & Archive Audit

**Document Version:** 1.0  
**Date:** 2026-11-04  
**Project:** MMAct Multimodal Human Activity Recognition  
**Audit Script:** `scripts/audit_dataset.py`  

---

## 1. Executive Summary

This document details the file-level availability, structure, modality specifications, and archive status of the MMAct dataset in workspace `D:\HAR_MMAct`.

### Key Findings:
1. **Challenge Codebase & Ground Truth:** The official challenge repository is present and completely intact under `challenge-main\challenge-main\`. It includes 6 validation ground-truth and sample submission files covering 35–37 activity classes.
2. **Local Video Assets:** Exactly 1 sample video clip is present locally:
   - `challenge-main\challenge-main\utils\sample_clip\carrying.mp4` (3.54 MB, 1920×1080 @ 30 fps, H.264, 120 frames / 4.0 seconds).
3. **Local Sensor Assets:** 0 sensor CSV files are currently present in the workspace.
4. **Local Archive Files:** 0 zip/tar archives are currently present in `D:\HAR_MMAct` or user download folders.
5. **Download Archives Requirement:** The full dataset is hosted externally on Box (~77–92 GB for video archives) and accessible via a request form. To satisfy our strict 2-day deadline without waiting for a 90 GB download, our pipeline must support:
   - A **Small-Data / Smoke-Test Mode** with synthetic or localized paired mini-batches to verify the complete end-to-end multi-modal architecture immediately.
   - A **Configurable Full-Data Mode** that ingests the actual MMAct directory layout when archives are provided.

---

## 2. Inventory of Available Files in Workspace

### A. Video & Media
| File Path | Size | Dimensions / Format | Description |
|---|---|---|---|
| `challenge-main\challenge-main\utils\sample_clip\carrying.mp4` | 3,710,688 bytes (~3.54 MB) | 1920×1080, 30.0 fps, 120 frames | Official sample trimmed clip for action `carrying` |

### B. Evaluation Ground Truth & Submission Schemas
| File Path | Size | Records / Classes | Task & Split Coverage |
|---|---|---|---|
| `challenge-main\challenge-main\evaluation\trimmed_val_scene_gt.json` | 666,021 bytes | 5,027 samples / 35 classes | Task 1: Trimmed Action Recognition (Cross-Scene validation split) |
| `challenge-main\challenge-main\evaluation\trimmed_val_scene_sample_format.json` | 781,638 bytes | 5,027 sample predictions | Submission format specification for cross-scene validation |
| `challenge-main\challenge-main\evaluation\trimmed_val_view_gt.json` | 460,021 bytes | 3,500 samples / 36 labels | Task 1: Trimmed Action Recognition (Cross-View validation split) |
| `challenge-main\challenge-main\evaluation\trimmed_val_view_sample_format.json` | 540,517 bytes | 3,500 sample predictions | Submission format specification for cross-view validation |
| `challenge-main\challenge-main\evaluation\untrimmed_val_gt.json` | 1,269,964 bytes | 315 sessions / 37 classes | Task 2: Untrimmed Temporal Action Localization |
| `challenge-main\challenge-main\evaluation\untrimmed_val_sample_submission.json` | 1,598,224 bytes | 315 session predictions | Submission format specification for untrimmed temporal localization |

### C. Reference Scripts & Technical Reports
| File Path | Purpose |
|---|---|
| `challenge-main\challenge-main\utils\time_series_classifiers.py` | Reference data loading (`datetime_merging`, window slicing) and InceptionTime sensor baseline |
| `challenge-main\challenge-main\utils\pose_frame_extraction.py` | OpenCV video frame iteration template |
| `challenge-main\challenge-main\evaluation\eval_mmact_trimmed.py` | Official evaluation script for Task 1 (classification mAP) |
| `challenge-main\challenge-main\evaluation\eval_mmact_untrimmed.py` | Official evaluation script for Task 2 (temporal localization AP) |
| `challenge-main\challenge-main\assets\report\OPPO_mmact_challenge_report.pdf` | Technical report detailing 3D CNN ensemble (CSN, SlowFast, TPN) |
| `challenge-main\challenge-main\assets\report\DeepblueAI-Challenge report for MMAct Challenge.pdf` | Technical report detailing TSM and ir-CSN action recognition |
| `challenge-main\challenge-main\assets\report\Raphael_unified_action_recognition_with_various_modalities.pdf` | Technical report detailing multi-modal 2D-CNN representation |

---

## 3. Sensor Modalities, Channels & Sampling Specifications

From `utils/time_series_classifiers.py` and the ICCV 2019 paper:

| Sensor Type | Folder Name | Device / Location | Sampling Rate | Columns / Channels | Preprocessing Applied in Baseline |
|---|---|---|---|---|---|
| **Phone Accelerometer** | `acc_phone_clip` | ASUS ZenPhone AR in front right pants pocket | 100 Hz | `time`, `x`, `y`, `z` | Suffix `s_acc_*`; values divided by 9.8 (convert m/s² to g's) |
| **Watch Accelerometer** | `acc_watch_clip` | Smartwatch on right wrist | 100 Hz | `time`, `x`, `y`, `z` | Suffix `w_acc_*`; values divided by 9.8 (convert m/s² to g's) |
| **Gyroscope** | `gyro_clip` | Smartphone in front right pants pocket | 50 Hz | `time`, `x`, `y`, `z` | Suffix `gyro_*`; resampled/merged to match 100 Hz timeline |
| **Orientation** | `orientation_clip` | Smartphone in front right pants pocket | 50 Hz | `time`, `azimuth`, `pitch`, `roll` | Angles in degrees; optional 3 channels |

### Sensor Data Alignment:
* Timestamp format in raw CSV files: `YYYYMMDD_HH:MM:SS.fff` (Timezone: `Asia/Tokyo`).
* `datetime_merging`: Uses the 100 Hz phone or watch accelerometer dataframe as primary index, left-joins secondary sensors (gyroscope, orientation), applies backward-fill (`bfill`) to align disparate sampling frequencies (50 Hz -> 100 Hz), and creates fixed 1.0-second sliding windows (100 samples) with 75% overlap (stride 25 samples).
* Total combined sensor channels:
  - **Minimal paired configuration:** 6 channels (`s_acc_x`, `s_acc_y`, `s_acc_z`, `w_acc_x`, `w_acc_y`, `w_acc_z`).
  - **Standard paired configuration:** 9 channels (6 accelerometer channels + `gyro_x`, `gyro_y`, `gyro_z`).
  - **Full sensor configuration:** 12 channels (9 IMU channels + 3 orientation channels).

---

## 4. Expected Dataset Directory Structure for Ingestion

When dataset archives are provided, our indexer will expect the following structure:

```text
<DATASET_ROOT>/
├── trimmed/
│   ├── video/
│   │   └── <split>/trainval/subject<ID>/cam<ID>/scene<ID>/session<ID>/<action>.mp4
│   └── sensors/
│       ├── acc_phone_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
│       ├── acc_watch_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
│       ├── gyro_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
│       └── orientation_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
```

---

## 5. Pipeline Architecture for Phase 2 Implementation

To ensure non-blocking development under the strict 2-day timeline:

1. **Dual-Mode Dataset Loader (`src/dataset/`):**
   * **Full-Data Mode:** Reads genuine files from `<DATASET_ROOT>` using path templates matching `trimmed/video` and `trimmed/sensors`.
   * **Small-Data / Smoke-Test Mode:**
     - Uses `carrying.mp4` and creates controlled mock clips and aligned multi-channel CSV sensor logs for the 10 verified target classes.
     - Enables immediate, reproducible validation of video decoding, sensor windowing, feature fusion, model forward/backward passes, checkpoint saving, and metrics generation.
2. **Subject-Independent Splitter (`src/dataset/splits.py`):**
   * Partitions by `subject` (e.g. subjects 1–14 Train, 15–17 Val, 18–20 Test) or `scene` (scenes 1–3 Train, scene 4 Val/Test) to prevent data leakage across train and evaluation sets.
3. **Preprocessing Cache (`src/dataset/cache.py`):**
   * Caches sampled video tensor representations and normalized sensor windows to `.pt` / `.npy` files to prevent repeated video decoding overhead during training.

---

## 6. Actionable Next Steps & Prerequisites

1. Implement `src/` modular package structure (preserving `challenge-main` untouched).
2. Implement Phase 2 dataset indexer, preprocessors, and small-data generator for rapid verification.
3. Validate Python runtime requirements (`torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `opencv-python`).
