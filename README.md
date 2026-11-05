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

### A. Run Unit Tests (33 Tests)
Validates indexer pairing, symlink traversal, label mapping, model architectures, preprocessing tensor shapes, zero-leakage split verification, hyperparameter tuning, and notebook structure:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" -m pytest tests -v
```

### B. Hyperparameter Tuning (`scripts/tune.py`)
Explores learning rate, weight decay, and dropout regularization strictly on train/val splits (test set remains strictly untouched):
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/tune.py `
    --data_root data/genuine_mmact `
    --split_strategy subject `
    --epochs_per_trial 10 `
    --batch_size 8 `
    --output_dir outputs/tuning `
    --plots_dir outputs/plots
```
Outputs:
- Trial history: `outputs/tuning/tuning_history.json`, `outputs/tuning/tuning_history.csv`, `outputs/tuning/trial_epoch_histories.csv`
- Best configuration: `outputs/tuning/best_hyperparameters.json`
- Visualization: `outputs/plots/hyperparameter_trials.png`

### C. Train Models (`scripts/train.py`)

#### 1. Pipeline Verification on Synthetic Smoke-Test:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/train.py --smoke_test --epochs 5 --batch_size 8
```

#### 2. Improved Custom CNN Baseline (50 Epochs):
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/train.py `
    --data_root data/genuine_mmact `
    --video_backbone cnn `
    --sensor_backbone inception_time `
    --mode video_only `
    --epochs 50 `
    --min_epochs 50 `
    --batch_size 8 `
    --lr 0.0001 `
    --weight_decay 0.001 `
    --dropout 0.35 `
    --split_strategy subject `
    --early_stopping_patience 15 `
    --checkpoint_name best_model_cnn.pth `
    --output_dir outputs/baseline_cnn
```

#### 3. MobileNetV2 Staged Fine-Tuning (50 Epochs):
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/train.py `
    --data_root data/genuine_mmact `
    --video_backbone mobilenet_v2 `
    --sensor_backbone inception_time `
    --mode video_only `
    --epochs 50 `
    --min_epochs 50 `
    --batch_size 8 `
    --lr 0.0001 `
    --weight_decay 0.0005 \\\
    --dropout 0.35 `
    --staged_finetune `
    --unfreeze_epoch 15 `
    --unfreeze_lr 0.00001 `
    --split_strategy subject `
    --early_stopping_patience 15 `
    --checkpoint_name best_model_mobilenet_v2.pth `
    --output_dir outputs/mobilenet_v2
```

#### 4. Sensor-Only InceptionTime Baseline (50 Epochs):
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/train.py `
    --data_root data/genuine_mmact `
    --mode sensor_only `
    --sensor_backbone inception_time `
    --epochs 50 `
    --min_epochs 50 `
    --batch_size 8 `
    --lr 0.0005 `
    --weight_decay 0.0001 `
    --split_strategy subject `
    --checkpoint_name best_model_sensor.pth `
    --output_dir outputs/sensor_only
```

#### 5. Multimodal Late Fusion (50 Epochs):
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/train.py `
    --data_root data/genuine_mmact `
    --mode multimodal `
    --video_backbone mobilenet_v2 `
    --sensor_backbone inception_time `
    --epochs 50 `
    --min_epochs 50 `
    --batch_size 8 `
    --lr 0.0001 `
    --weight_decay 0.0005 `
    --split_strategy subject `
    --checkpoint_name best_model_multimodal.pth `
    --output_dir outputs/multimodal_fusion
```

### D. Standalone Held-Out Test Evaluation (`scripts/evaluate.py`)
Computes side-by-side performance metrics for **Multimodal (Fused)**, **Video-Only**, and **Sensor-Only** baselines:
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/evaluate.py `
    --checkpoint outputs/multimodal_fusion/checkpoints/best_model_multimodal.pth `
    --data_root data/genuine_mmact `
    --output_file outputs/multimodal_fusion/metrics.json `
    --plots_dir outputs/multimodal_fusion/plots
```
Outputs:
- Comprehensive metrics: `metrics.json` and `metrics.csv`
- Detailed sample predictions: `test_predictions.csv`
- Test confusion matrix: `test_confusion_matrix.png`
- Test One-vs-Rest ROC curves (with per-class, micro, and macro AUC): `test_roc_ovr.png`
- Test Precision-Recall curves (with per-class, micro, and macro AP): `test_pr_curves.png`
- Standalone per-class bar charts: `test_precision_bars.png`, `test_recall_bars.png`, `test_f1_bars.png`
- Cross-modality ablation bar chart: `test_modality_ablation.png`

### E. Single-Sample / Paired Inference (`scripts/infer.py`)

#### 1. Multimodal Inference (Video + Sensors):
```powershell
& "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" scripts/infer.py `
    --video path/to/clip.mp4 `
    --sensor_phone path/to/acc_phone.csv `
    --sensor_watch path/to/acc_watch.csv `
    --sensor_gyro path/to/gyro.csv `
    --checkpoint outputs/multimodal_fusion/checkpoints/best_model_multimodal.pth
```

---

## 7. Model Architectures & Benchmark Suite

### A. Improved Custom CNN (Anti-Overfitting Regularization)
- **Architecture**: 3-stage hierarchical 2D convolutional blocks with Spatial Dropout (`Dropout2d(0.15–0.20)`), Batch Normalization after every convolution, and Global Average Pooling per frame.
- **Regularization**: High temporal dropout ($p=0.35–0.40$), $L_2$ weight decay ($1\times 10^{-3}$), Cosine Annealing LR scheduling, and best validation checkpointing.
- **Temporal Aggregation**: Concatenated temporal Mean + Max pooling across frame representations ($128 \times 2 = 256$ dims) projected to a 128-dim video embedding.

### B. MobileNetV2 (Transfer Learning & Staged Fine-Tuning)
- **Architecture**: Inverted residual blocks with linear bottlenecks (`torchvision` ImageNet pretrained backbone with pure PyTorch fallback for zero-dependency execution).
- **Staged Fine-Tuning Strategy**:
  - Phase 1 (Epochs 1 to 15): Feature extraction backbone frozen (`requires_grad = False`), training projection and classification heads with LR $1\times 10^{-4}$.
  - Phase 2 (Epochs 16 to 50): Entire backbone unfrozen with differential learning rates (backbone LR $1\times 10^{-5}$, classification heads LR $1\times 10^{-4}$).

### C. InceptionTime Sensor Encoder & Multimodal Late Fusion
- **Sensor Branch**: Multi-scale 1D InceptionTime modules (kernel sizes 9, 19, 39, residual shortcuts) extracting temporal dynamics across 9 inertial channels (`acc_phone`, `acc_watch`, `gyro`).
- **Multimodal Fusion**: Concatenates 128-dim video embedding and 128-dim sensor embedding ($256$ dims) $\rightarrow$ Dropout ($0.35$) $\rightarrow$ Linear ($128$) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout ($0.35$) $\rightarrow$ Linear ($10$ classes).

---

## 8. Kaggle Cloud GPU Execution

For seamless cloud execution on Kaggle GPU, three notebooks are available in `notebooks/`:
- **Complete Fast Notebook (Recommended)**: [`notebooks/kaggle_mmact_har_complete_fast.ipynb`](notebooks/kaggle_mmact_har_complete_fast.ipynb) — Featuring shallow bounded discovery (0.3s), idempotent staging with preflight report, zero duplicate links, auto-resume, skip-if-completed, AMP acceleration, and optional extended benchmarks.
- **50-Epoch Benchmark Notebook**: [`notebooks/kaggle_mmact_har_benchmark_50ep.ipynb`](notebooks/kaggle_mmact_har_benchmark_50ep.ipynb)
- **Standard Pipeline Notebook**: [`notebooks/kaggle_mmact_har.ipynb`](notebooks/kaggle_mmact_har.ipynb)

All notebooks run completely top-to-bottom without manual intervention:
1. **Upload Notebook to Kaggle**: Import `notebooks/kaggle_mmact_har_complete_fast.ipynb` as a new Kaggle Notebook.
2. **Configure Session Options**:
   - **Accelerator**: `GPU T4 x2` or `GPU P100`
   - **Internet**: `On` (required to sync repository and pretrained weights)
3. **Attach MMAct Dataset**:
   - Click **+ Add Input** in the top right.
   - Attach your uploaded MMAct dataset (e.g. `mmact-trimmed`).
4. **Automated Pipeline Steps**:
   - **Step 1**: CUDA GPU diagnostics and memory verification.
   - **Step 2**: Clones/updates `master` branch and verifies NumPy/Pandas C ABI preservation (`requirements-kaggle.txt`).
   - **Step 3**: Discovers and ingests all video subjects, sensor modalities, and split files across all attached inputs (following symlinks with recursion loop protection).
   - **Step 4**: Pre-flight readability audit (decoding sample video frame, parsing sensor CSV, verifying split isolation with zero leakage).
   - **Step 5**: Preprocessing & tensor caching warm-up.
   - **Step 6**: Experiment 1 – Custom CNN Baseline (50 epochs, spatial dropout, batchnorm, weight decay).
   - **Step 7**: Experiment 2 – Hyperparameter Tuning (train/val only) & Tuned CNN (50 epochs on optimal configuration).
   - **Step 8**: Experiment 3 – MobileNetV2 Staged Fine-Tuning (50 epochs: 15 epochs frozen -> 35 epochs unfrozen).
   - **Step 9**: Experiment 4 & 5 – Sensor-Only InceptionTime (50 epochs) & Multimodal Late Fusion (50 epochs).
   - **Step 10**: Standalone held-out test evaluation across all 5 models.
   - **Step 11**: Generation and display of all publication plots and comparison galleries.
   - **Step 12**: Interactive sample multimodal inference demo.
   - **Step 13**: Consolidated results summary table and 1-click artifact packaging (`har_mmact_results.zip`).

---

## 9. Known Limitations & Integrity Commitments

1. **Dataset Availability:**
   - The workspace contains `test_set_cross-view_1.zip` (14.99 GB, 13,869 unlabelled `.mp4` video files with obfuscated names for the CodaLab challenge leaderboard).
   - This test release contains no ground truth labels and no sensor CSV files.
   - Final official benchmark metrics on genuine paired MMAct data require downloading the trimmed train/val archives from the official Box repository (password authorized via the challenge organizers).
2. **Zero Fabrication:**
   - All synthetic smoke test artifacts are explicitly flagged with `is_synthetic: true` and `[SYNTHETIC]` metadata in `outputs/metrics.json` and `outputs/training_summary.json`.
   - Synthetic pipeline runs are never reported as genuine MMAct evaluation results.
