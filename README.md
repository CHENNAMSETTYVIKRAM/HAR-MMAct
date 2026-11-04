# HAR_MMAct: Multimodal Human Activity Recognition

A reproducible, software-only multimodal Human Activity Recognition (HAR) framework pairing video clips and wearable inertial sensor streams from the [MMAct dataset](https://mmact19.github.io/challenge/).

---

## 1. Verified Target Activity Classes

Cross-referenced and verified against the official challenge validation ground truth files (`trimmed_val_scene_gt.json`, `trimmed_val_view_gt.json`) and the ICCV 2019 MMAct publication (see [`docs/LABEL_MAPPING.md`](docs/LABEL_MAPPING.md)):

| Class Index | Target Class | Modality Signals | Source File Key |
|---|---|---|---|
| `0` | `crouching` | Elevation drop, postural change | `crouching` |
| `1` | `jumping` | Sharp vertical acceleration peaks | `jumping` |
| `2` | `kicking` | Leg acceleration (`acc_phone`) | `kicking` |
| `3` | `pointing` | Wrist acceleration (`acc_watch`) | `pointing` |
| `4` | `running` | High-cadence periodic locomotion | `running` |
| `5` | `sitting` | Static seated posture | `sitting` |
| `6` | `standing` | Static upright posture | `standing` |
| `7` | `using_phone` | Wrist/hand manipulation | `using_phone` |
| `8` | `walking` | Regular gait locomotion | `walking` |
| `9` | `waving_hand` | Periodic wrist oscillation | `waving_hand` |

---

## 2. Model Architecture

The framework supports multimodal fusion and standalone unimodal evaluation:

1. **Video Branch (`ResNet18VideoBranch`):**
   - Pure PyTorch implementation of ResNet-18 without external C++ binary dependencies.
   - Loads pretrained ImageNet weights from `outputs/pretrained/resnet18.pth` (46.8 MB).
   - Frame sampling: 16 frames uniformly sampled per clip, resized to 112×112.
   - Temporal aggregation: Dual Mean + Max pooling across temporal representations yielding 1024-dim clip features, projected to 128-dim embedding.
   - Lightweight alternative (`LightweightVideoBranch`) also supported for fast execution.

2. **Sensor Branch (`InceptionTimeSensorBranch`):**
   - 1D InceptionTime architecture matching official MMAct challenge conventions (Fawaz et al., 2020).
   - Multi-scale 1D convolutions with receptive field kernel sizes 9, 19, 39, bottleneck projection (32 filters), and residual shortcut connections.
   - Global average pooling followed by projection to 128-dim embedding.
   - Lightweight alternative (`Sensor1DCNNBranch`) also available.

3. **Multimodal Fusion (`MultimodalHARModel`):**
   - Concatenates 128-dim video embedding and 128-dim sensor embedding into a 256-dim fused feature vector.
   - Multi-layer perceptron (Linear + BatchNorm1d + ReLU + Dropout) outputs 10-class activity logits.
   - Includes dedicated auxiliary classification heads for isolated **video-only** and **sensor-only** baseline benchmarking.

---

## 3. Directory Layout

```text
D:\HAR_MMAct\
├── challenge-main\            # Official challenge repository (preserved unmodified)
├── docs\
│   ├── INITIAL_RESOURCE_AUDIT.md # Repository, license, and asset inventory
│   ├── LABEL_MAPPING.md          # 10 verified target classes & disambiguation
│   ├── DATASET_AUDIT.md          # Sensor channels, sampling rates, layout
│   └── PHASE2_REPORT.md          # Pipeline architecture & unit test reports
├── outputs\
│   ├── checkpoints\              # Trained model checkpoints (.pth)
│   ├── plots\                    # Loss/accuracy curves, confusion matrices, ROC/PR
│   ├── pretrained\               # Pretrained backbones (resnet18.pth)
│   ├── cache\                    # Preprocessed tensor cache (.pt)
│   ├── metrics.json              # Detailed evaluation metrics
│   └── training_summary.json     # Training run summary and hyperparameter metadata
├── scripts\
│   ├── train.py                  # Configurable training CLI
│   ├── evaluate.py               # Standalone evaluation and unimodal comparison
│   ├── infer.py                  # Inference CLI (multimodal, video-only, sensor-only)
│   ├── audit_dataset.py          # Ground truth audit tool
│   ├── generate_synthetic_data.py# Synthetic smoke-test generator
│   └── run_smoke_test.py         # End-to-end pipeline validation
├── src\
│   ├── config.py                 # Seeds, paths, sliding window, model hyperparameters
│   ├── data\
│   │   ├── indexer.py            # Multimodal sample pairing & anomaly reporting
│   │   ├── video_loader.py       # Frame extraction, uniform sampling, resize
│   │   ├── sensor_loader.py      # CSV parsing, g-scaling, resampling, windowing
│   │   ├── splits.py             # Leakage-free subject, view, scene, session splits
│   │   ├── dataset.py            # PyTorch Dataset with on-disk tensor caching
│   │   └── synthetic.py          # Synthetic fixture generator with [SYNTHETIC] metadata
│   ├── models\
│   │   ├── video_branch.py       # Pretrained ResNet-18 & Lightweight 2D CNN
│   │   ├── sensor_branch.py      # InceptionTime & 1D CNN
│   │   └── fusion_model.py       # Multimodal fusion & unimodal baseline heads
│   └── utils\
│       ├── metrics.py            # Accuracy, Macro F1, Precision, Recall, ROC AUC, AP
│       └── plots.py              # Publication-ready visualizations
├── tests\
│   ├── test_indexer.py           # Pairing and anomaly detection tests
│   ├── test_label_mapping.py     # Class mapping verification tests
│   ├── test_models.py            # ResNet-18, InceptionTime, fusion forward/backward tests
│   ├── test_preprocessing.py     # Video and sensor tensor shape tests
│   └── test_splits.py            # Zero-leakage split isolation tests
├── requirements.txt
└── README.md
```

---

## 4. Environment Setup

The codebase runs on Python 3.11 with standard PyTorch (CPU or CUDA).

```powershell
# Using Python 3.11
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt
```

### Pretrained Backbone Download
The ResNet-18 backbone weights are pre-cached in `outputs/pretrained/resnet18.pth` (46.8 MB). If downloading freshly:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" -c "import urllib.request; urllib.request.urlretrieve('https://download.pytorch.org/models/resnet18-f37072fd.pth', 'outputs/pretrained/resnet18.pth')"
```

---

## 5. Genuine Data Placement Guide

To ingest the genuine MMAct dataset:

1. Download the trimmed archives from the official MMAct Box repository (authorized via the challenge Google Form).
2. Place genuine trimmed videos and sensors under `data/genuine_mmact/` matching the standard layout:
   ```text
   data/genuine_mmact/
   └── trimmed/
       ├── video/
       │   ├── cross_scene/trainval/subject<ID>/cam<ID>/scene<ID>/session<ID>/<activity>.mp4
       │   └── cross_view/trainval/subject<ID>/cam<ID>/scene<ID>/session<ID>/<activity>.mp4
       └── sensors/
           ├── acc_phone_clip/subject<ID>/scene<ID>/session<ID>/<activity>.csv
           ├── acc_watch_clip/subject<ID>/scene<ID>/session<ID>/<activity>.csv
           └── gyro_clip/subject<ID>/scene<ID>/session<ID>/<activity>.csv
   ```
3. If disk space is limited, extract a balanced subset of subjects (e.g. subjects 1 to 5) across the 10 target classes. The indexer will pair available files and filter unsupported labels.

---

## 6. How to Run

### A. Run Unit Tests (16 Tests)
Validates indexer pairing, label mapping, model architectures, preprocessing tensor shapes, and zero-leakage split verification:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" -m pytest tests -v
```

### B. Train Model (`scripts/train.py`)

#### Pipeline Verification on Synthetic Fixture:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/train.py --smoke_test --epochs 5 --batch_size 8
```

#### Training on Genuine MMAct Data:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/train.py `
    --data_root data/genuine_mmact `
    --epochs 50 `
    --batch_size 16 `
    --lr 0.0001 `
    --split_strategy subject `
    --video_backbone resnet18 `
    --sensor_backbone inception_time `
    --mode multimodal
```

Key flags:
- `--mode`: `multimodal` (default), `video_only`, `sensor_only`
- `--split_strategy`: `subject` (default), `cross_view`, `cross_scene`, `session`
- `--video_backbone`: `resnet18` (default), `lightweight`
- `--sensor_backbone`: `inception_time` (default), `cnn1d`
- `--freeze_video`: Freezes pretrained video backbone for fast sensor-focused fine-tuning

Outputs:
- Checkpoints: `outputs/checkpoints/best_model.pth`, `outputs/checkpoints/latest_model.pth`
- Visualizations: `outputs/plots/training_curves.png`, `outputs/plots/val_confusion_matrix.png`, `outputs/plots/val_roc_ovr.png`, `outputs/plots/val_pr_curves.png`
- Summary metadata: `outputs/training_summary.json`

### C. Standalone Evaluation (`scripts/evaluate.py`)
Computes side-by-side performance metrics for **Multimodal (Fused)**, **Video-Only**, and **Sensor-Only** baselines:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```
Outputs:
- Comprehensive metrics: `outputs/metrics.json`
- Test confusion matrix: `outputs/plots/test_confusion_matrix.png`
- Test One-vs-Rest ROC curves: `outputs/plots/test_roc_ovr.png`
- Test Precision-Recall curves: `outputs/plots/test_pr_curves.png`

### D. Single-Sample / Paired Inference (`scripts/infer.py`)

#### 1. Multimodal Inference (Video + Sensors):
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/infer.py `
    --video path/to/clip.mp4 `
    --sensor_phone path/to/acc_phone.csv `
    --sensor_watch path/to/acc_watch.csv `
    --sensor_gyro path/to/gyro.csv
```

#### 2. Video-Only Inference:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/infer.py --video path/to/clip.mp4
```

#### 3. Sensor-Only Inference:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/infer.py `
    --sensor_phone path/to/acc_phone.csv `
    --sensor_watch path/to/acc_watch.csv `
    --sensor_gyro path/to/gyro.csv
```

---

## 7. Known Limitations & Integrity Commitments

1. **Dataset Availability:**
   - The workspace contains `test_set_cross-view_1.zip` (14.99 GB, 13,869 unlabelled `.mp4` video files with obfuscated names for the CodaLab challenge leaderboard).
   - This test release contains no ground truth labels and no sensor CSV files.
   - Final official benchmark metrics on genuine paired MMAct data require downloading the trimmed train/val archives from the official Box repository (password authorized via the challenge organizers).
2. **Zero Fabrication:**
   - All synthetic smoke test artifacts are explicitly flagged with `is_synthetic: true` and `[SYNTHETIC]` metadata in `outputs/metrics.json` and `outputs/training_summary.json`.
   - Synthetic pipeline runs are never reported as genuine MMAct evaluation results.
