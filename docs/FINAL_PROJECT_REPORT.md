# HAR-MMAct: Multimodal Human Activity Recognition - Final Project Report

**Document Version:** 1.1
**Date:** 2026-11-16
**Repository:** `https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct`
**Current Branch:** `master`
**Project Workspace:** `D:\HAR_MMAct`

---

## 1. Executive Summary

This report documents the design, implementation, validation benchmark, and held-out test evaluation of the **HAR-MMAct** software-only multimodal Human Activity Recognition system. The framework combines video clips and wearable inertial sensor time series from the genuine MMAct dataset and targets 10 verified activity classes.

### Verified Accomplishments and Results

1. **Multimodal data pipeline:** Implemented video and sensor ingestion, preprocessing, sample pairing, and sensor temporal alignment.
2. **Validation benchmark:** The selected `exp1_multimodal` model (MobileNetV2 + InceptionTime late fusion) achieved 81.15% validation accuracy and 0.8058 validation Macro F1 on 780 samples.
3. **Independent modality evaluations:** Video-only MobileNetV2 and sensor-only InceptionTime models were evaluated alongside the multimodal model.
4. **Held-out test comparison:** All three models were evaluated on the same 852 subject-independent test samples. Their prediction CSVs were verified to contain identical `sample_id` values in identical order.

| Model | Test Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---:|---:|---:|---:|
| Video-only MobileNetV2 | 42.49% | 0.4317 | 0.4665 | 0.4552 |
| Sensor-only InceptionTime | 69.95% | 0.6665 | 0.7102 | 0.6987 |
| **Multimodal MobileNetV2 + InceptionTime** | **76.76%** | **0.7545** | **0.7843** | **0.7652** |

5. **Multimodal ranking:** The multimodal model achieved the strongest overall held-out test performance among these three evaluated configurations.
6. **Reproducibility artifacts:** Evaluation JSON files, prediction CSVs, per-class metrics, and comparison plots were generated and preserved in the repository output structure.

The results are specific to this dataset, split, and experimental setup. They do not establish performance on unseen datasets or deployment conditions.

---

## 2. Dataset Sourcing & Kaggle Downloads

The project utilizes trimmed video and wearable sensor streams from the MMAct Human Activity Recognition benchmark. To support reproducible training and cloud GPU execution without requiring external Box API access, the dataset assets are hosted on Kaggle:

### Kaggle Datasets:
1. **Trimmed Scene Video Dataset:**
   [https://www.kaggle.com/datasets/vikramche/trimmed-scene-video](https://www.kaggle.com/datasets/vikramche/trimmed-scene-video)
   *Description:* Kaggle-hosted video dataset prepared for the project's video modality, organized by subject, camera, scene, and session.
2. **MMAct Trimmed Dataset:**
   [https://www.kaggle.com/datasets/vikramche/mmact-trimmed](https://www.kaggle.com/datasets/vikramche/mmact-trimmed)
   *Description:* Kaggle-hosted trimmed MMAct dataset containing 4 sensor modalities (`acc_phone_clip`, `acc_watch_clip`, `gyro_clip`, `orientation_clip`), official split configuration files, and baseline trimmed clips.

> [!NOTE]
> These Kaggle datasets are community-mirrored for project reproducibility and are separate from the Git repository. Raw video and sensor archives are not committed to Git.

### Expected Local Dataset Layout
When downloaded and extracted locally or staged on Kaggle, the data loader expects the following canonical directory structure under `data/genuine_mmact/` (or `/kaggle/working/data/genuine_mmact/`):

```text
data/genuine_mmact/
-- trimmed/
    |-- video/
    |   -- cross_scene/
    |       -- trainval/
    |           -- subject<ID>/
    |               -- cam<ID>/
    |                   -- scene<ID>/
    |                       -- session<ID>/
    |                           -- <action>.mp4
    |-- sensors/
    |   |-- acc_phone_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
    |   |-- acc_watch_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
    |   |-- gyro_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
    |   -- orientation_clip/subject<ID>/scene<ID>/session<ID>/<action>.csv
    -- splits/
        |-- trimmed_x-scene_train.txt
        |-- trimmed_x-scene_val.txt
        |-- trimmed_x-view_train.txt
        -- trimmed_x-view_val.txt
```

---

## 3. Verified Target Classes & Label Mapping

The project focuses on 10 strictly verified target physical activity classes, cross-referenced against the official MMAct validation ground truth files (`trimmed_val_scene_gt.json`, `trimmed_val_view_gt.json`):

| Class Index | Canonical Target Class | Primary Physical Cues | Inertial Signals |
|:---:|---|---|---|
| `0` | `crouching` | Downward vertical displacement | Sudden negative acceleration on vertical axis |
| `1` | `jumping` | Rapid upward thrust & landing impact | High-amplitude acceleration spikes (`acc_phone`) |
| `2` | `kicking` | Rapid unilateral leg extension | High jerk on thigh/pocket accelerometer |
| `3` | `pointing` | Sustained directional arm extension | Static pitch/roll reorientation on smartwatch |
| `4` | `running` | Fast periodic locomotion | High-frequency oscillatory gait cadence |
| `5` | `sitting` | Static seated posture | Low-variance stable gravity component on thigh |
| `6` | `standing` | Static upright posture | Stable vertical gravity vector ($\approx 9.8\text{ m/s}^2$) |
| `7` | `using_phone` | Hand manipulation near body | Fine rotational changes on wrist/watch |
| `8` | `walking` | Moderate periodic locomotion | Standard 1–2 Hz gait acceleration cycles |
| `9` | `waving_hand` | Cyclic wrist oscillation | High-frequency periodic angular velocity on gyro |

---

## 4. Ingestion, Preprocessing & Alignment Methodology

### 4.1 Video Preprocessing Pipeline
- **Sampling Strategy:** Uniform temporal subsampling of $T=16$ frames per video clip using OpenCV (`cv2.VideoCapture`).
- **Spatial Resolution:** Resized to $112 \times 112$ pixels using bilinear interpolation.
- **Normalization:** Channel-wise standardization using standard ImageNet parameters:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Output Tensor:** Dimensioned as $(B, 3, 16, 112, 112)$ for 3D/temporal models, reshaped dynamically to $(B \times 16, 3, 112, 112)$ for 2D feature extraction.

### 4.2 Sensor Ingestion & Timestamp Alignment
Sensor processing follows the official MMAct challenge methodology ([`challenge-main/challenge-main/utils/time_series_classifiers.py`](file:///d:/HAR_MMAct/challenge-main/challenge-main/utils/time_series_classifiers.py)):
1. **Unit Conversion:** Accelerometer data is converted from $\text{m/s}^2$ to $g$ units by dividing by $9.8$.
2. **Uniform Timegrid Generation:**
   $$\text{time\_values}[i] = \text{round}\left(1000 \times \left(\text{stt\_time\_sec} + \frac{i}{\text{sample\_rate}}\right)\right)$$
   Generates uniform integer millisecond timegrids (10 ms for 100 Hz phone/watch accelerometers; 20 ms for 50 Hz gyroscope), eliminating hardware sampling jitter while preserving all recorded rows.
3. **Primary Timeline & Left Join:** The 100 Hz phone accelerometer serves as the primary index; watch accelerometer and gyroscope dataframes are left-joined onto this timeline.
4. **Backward-Fill Resampling:** Intermediate 100 Hz timestamps between 50 Hz samples receive the upcoming measurement via backward-fill (`bfill()`).
5. **Boundary & Gap Safeguards:** Trailing rows where secondary sensors ceased recording are trimmed via `dropna()`. Optional bounded forward fill (`ffill(limit=2)`) is strictly limited to at most 1 secondary cycle ($20\text{ ms} = 2\text{ samples}$), strictly preventing artificial flatline signals across large missing intervals.
6. **Fixed Window Extraction:** Centered sliding window of $W=100$ samples (1.0 second duration) is extracted, with symmetric edge-padding applied if a clip duration is shorter than 100 samples. Output tensor shape: $(B, 9, 100)$.

---

## 5. Split Protocol & Leakage Prevention

### Official MMAct Release vs. Project Partitioning

- **Official MMAct protocol:** The public release provides official train and validation split files. It does not provide a locally labelled official test partition.
- **Project subject-independent partition:** The experiments use a deterministic, subject-independent partition. Model training uses the training partition; model selection uses validation; the held-out test partition is kept separate from training and tuning.

### Partition Statistics

| Partition | Samples | Recordings | Use |
|---|---:|---:|---|
| Training | 3,795 | 595 | Model fitting |
| Validation | 780 | 124 | Model selection |
| Held-out test | 852 | 132 | Final evaluation |

The final test predictions from the video-only, sensor-only, and multimodal models were checked against actual MMAct `sample_id` values. All three files contained the same 852 unique IDs in identical order.

The split-isolation audit recorded no overlap of subjects, videos, recordings, or sessions between train, validation, and test partitions for the selected subject split.

---

## 6. Model Architectures & Fusion Strategies

```
                      ┌────────────────────────────────────────┐
                      |    Video Input: (B, 3, 16, 112, 112)   │
                      --─────────────────┬────────────────────┘
                                          │
                                   MobileNetV2
                               (ImageNet Pretrained)
                                          │
                              Frame Embeddings (1280-d)
                                          │
                                 Temporal Pooling
                                 (Mean + Max Pool)
                                          │
                             Video Feature Vector (256-d)
                                          │
    ┌─────────────────────────────────────┴─────────────────────────────────────┐
    |                                                                           │
    ▼                                                                           ▼
[Video Classifier Head]                                              ┌─────────────────────┐
  (Video-Only Mode)                                                  |   Late Fusion MLP   │
  Logits: (B, 10)                                                    |    Concatenation    │
                                                                     |     (256 + 256)     │
[Sensor Classifier Head]                                             |          |          │
  (Sensor-Only Mode)                                                 |    Fused Vector     │
  Logits: (B, 10)                                                    |       (512-d)       │
    ▲                                                                |          |          │
    |                                                                |    Linear (128)     │
    --───────────────────────────────────┬──────────────────────────┤    BatchNorm1d      │
                                          |                          |       ReLU          │
                             Sensor Feature Vector (256-d)           |   Dropout (0.35)    │
                                          |                          |          |          │
                                InceptionTime 1D CNN                 --────────┬──────────┘
                                (Bottleneck + ResNet)                           │
                                          |                                     ▼
                      ┌───────────────────┴────────────────────┐      [Multimodal Head]
                      |     Sensor Input: (B, 9, 100)          |       Logits: (B, 10)
                      --──────────────────────────────────────┘
```

1. **Multimodal Late Fusion (`exp1_multimodal` / `multimodal_fusion`):**
   - Video stream: MobileNetV2 extracts 1280-dim per-frame features $\rightarrow$ dual Mean+Max temporal pooling $\rightarrow$ Linear(256) projection.
   - Sensor stream: InceptionTime extracts multi-scale 1D temporal representations across 9 channels $\rightarrow$ global average pooling $\rightarrow$ Linear(256) projection.
   - Fusion: Concatenation (512-dim) $\rightarrow$ Linear(128) $\rightarrow$ BatchNorm1d $\rightarrow$ ReLU $\rightarrow$ Dropout(0.35) $\rightarrow$ Linear(10).
2. **Gated Multimodal Fusion (`tcn_gated_fusion`):**
   - Incorporates a learned gating mechanism where sensor confidence dynamically reweights video representations prior to classification.
3. **Sensor-Only TCN (`tcn_sensor_only`):**
   - Evaluates dilated causal 1D convolutions across the sensor stream.
4. **Sensor-Only InceptionTime (`exp2_sensor_only`):**
   - Evaluates the 9-channel inertial stream alone using InceptionTime with multi-scale kernel widths $(9, 19, 39)$ and residual shortcuts.
5. **MobileNetV2 Video-Only (`mobilenet_v2` & `mobilenet_v2_staged_v2`):**
   - Evaluates video modality alone using staged fine-tuning (backbone frozen during initial phase, then unfrozen with differential learning rates).
6. **Custom CNN Video-Only (`improved_cnn` & `improved_cnn_v2`):**
   - Scratch hierarchical 2D CNN with spatial dropouts, batch normalization, and global pooling.

---

## 7. Validation Benchmark Comparison & Model Ranking (9 Evaluated Models)

All 9 model configurations were evaluated on the identical validation split ($N=780$ samples) under standardized 50-epoch schedules on a Tesla T4 GPU with automatic mixed precision (AMP) and seed 42. Benchmark ranking is sorted in descending order of Validation Macro-F1 score:

### Complete Validation Benchmark Results:

| Rank | Model Experiment | Modality | Video Backbone | Sensor Backbone | Val Accuracy (%) | Val Macro F1 | Macro Precision | Macro Recall | Validation Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **`exp1_multimodal`** | **Multimodal** | **MobileNetV2** | **InceptionTime** | **81.15%** | **0.8058** | **0.8657** | **0.8066** | **Top Model (Selected)** |
| 2 | `multimodal_fusion` | Multimodal | MobileNetV2 | InceptionTime | 80.38% | 0.8005 | 0.8509 | 0.7987 | Robust Alternative |
| 3 | `tcn_gated_fusion` | Multimodal | MobileNetV2 | TCN 1D | 77.69% | 0.7643 | 0.8261 | 0.7711 | Gated Baseline |
| 4 | `tcn_sensor_only` | Sensor-Only | - | TCN 1D | 77.44% | 0.7538 | 0.7916 | 0.7684 | Top Sensor Baseline |
| 5 | `exp2_sensor_only` | Sensor-Only | - | InceptionTime | 77.44% | 0.7489 | 0.8219 | 0.7684 | Strong Unimodal |
| 6 | `mobilenet_v2_staged_v2` | Video-Only | MobileNetV2 | - | 48.59% | 0.5026 | 0.5519 | 0.5106 | Staged Fine-Tuned |
| 7 | `mobilenet_v2` | Video-Only | MobileNetV2 | - | 35.26% | 0.3492 | 0.4340 | 0.3900 | Under-adapted |
| 8 | `improved_cnn_v2` | Video-Only | Custom CNN | - | 16.03% | 0.1171 | 0.0951 | 0.2013 | Scratch Retrained |
| 9 | `improved_cnn` | Video-Only | Custom CNN | - | 14.74% | 0.0981 | 0.1251 | 0.1733 | Cold-Start Stalled |

### Key Benchmark Insights:
1. **Multimodal Superiority:** Combining visual cues with inertial motion dynamics provides an immediate **+3.71% accuracy increase** over the best sensor-only model and **+32.56% increase** over the best video-only model.
2. **Inertial Dominance in Ambiguous Poses:** Stationary and upright activities (`standing`, `sitting`, `crouching`) are heavily disambiguated by accelerometer gravity vectors, which compensate for camera perspective distortions.
3. **Video Transfer Learning Necessity:** Scratch CNNs failed to learn robust motion patterns on 3,795 clips (14.7–16.0% accuracy), while ImageNet-pretrained MobileNetV2 reached 48.59% accuracy when properly fine-tuned.

---

## 8. Final Held-Out Test Evaluation Results

Three model configurations were evaluated on the same 852-sample subject-independent held-out test partition. The test set was not used for training, hyperparameter tuning, or model selection.

### 8.1 Overall Test Metrics

| Metric | Video-only MobileNetV2 | Sensor-only InceptionTime | Multimodal Fusion |
|---|---:|---:|---:|
| Accuracy | 42.49% | 69.95% | **76.76%** |
| Macro F1 | 0.4317 | 0.6665 | **0.7545** |
| Macro Precision | 0.4665 | 0.7102 | **0.7843** |
| Macro Recall | 0.4552 | 0.6987 | **0.7652** |
| ROC-AUC (OvR) | 0.8443 | 0.9373 | **0.9661** |
| Macro Average Precision | 0.4732 | 0.7959 | **0.8590** |

### 8.2 Multimodal Per-Class Test Performance

| Activity Class | Support | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| `crouching` | 84 | 0.6754 | 0.9167 | 0.7778 |
| `jumping` | 84 | 0.3492 | 0.2619 | 0.2993 |
| `kicking` | 84 | 0.6351 | 0.5595 | 0.5949 |
| `pointing` | 84 | 1.0000 | 0.6786 | 0.8085 |
| `running` | 84 | 1.0000 | 0.4524 | 0.6230 |
| `sitting` | 48 | 0.9600 | 1.0000 | 0.9796 |
| `standing` | 84 | 0.6357 | 0.9762 | 0.7700 |
| `using_phone` | 132 | 0.8725 | 0.9848 | 0.9253 |
| `walking` | 84 | 0.8023 | 0.8214 | 0.8118 |
| `waving_hand` | 84 | 0.9130 | 1.0000 | 0.9545 |

### 8.3 Interpretation

- Sensor-only InceptionTime substantially outperformed video-only MobileNetV2 on this test partition.
- Multimodal fusion improved test accuracy by 6.81 percentage points over sensor-only and by 34.27 percentage points over video-only.
- The multimodal model achieved the highest Macro F1, ROC-AUC, and macro average precision of the three evaluated configurations.
- `jumping` remains the weakest multimodal class by F1 (0.2993). `running` has 0.4524 recall, indicating that many actual running samples were missed.
- Strong aggregate metrics should not obscure these class-specific weaknesses.

### 8.4 Test Artifact Locations

- Consolidated metrics: `outputs/model_comparison/test_metrics_comparison.csv`
- Per-class comparison: `outputs/model_comparison/per_class_test_metrics.csv`
- Accuracy graph: `outputs/model_comparison/test_accuracy_comparison.png`
- Macro F1 graph: `outputs/model_comparison/test_macro_f1_comparison.png`
- Per-class F1 graph: `outputs/model_comparison/per_class_test_f1_comparison.png`
- Multimodal evaluation: `outputs/exp1_multimodal/test_evaluation_v2.json`
- Sensor-only evaluation: `outputs/sensor_only_inception_time_v2/test_evaluation.json`
- Video-only evaluation: `outputs/mobilenet_v2_staged_v2_retrained/test_evaluation_v2.json`

---

## 9. Artifact Index

### Consolidated Model Comparison

| Artifact | Path | Purpose |
|---|---|---|
| Overall metrics CSV | `outputs/model_comparison/test_metrics_comparison.csv` | Test metrics for all three models |
| Overall metrics JSON | `outputs/model_comparison/test_metrics_comparison.json` | Machine-readable comparison |
| Per-class metrics CSV | `outputs/model_comparison/per_class_test_metrics.csv` | Per-class precision, recall, and F1 for each model |
| Accuracy graph | `outputs/model_comparison/test_accuracy_comparison.png` | Overall test accuracy comparison |
| Macro F1 graph | `outputs/model_comparison/test_macro_f1_comparison.png` | Overall test Macro F1 comparison |
| Per-class F1 graph | `outputs/model_comparison/per_class_test_f1_comparison.png` | Class-level F1 comparison |

### Per-Model Evaluation Files

| Model | Evaluation JSON | Prediction CSV | Plot Directory |
|---|---|---|---|
| Video-only MobileNetV2 | `outputs/mobilenet_v2_staged_v2_retrained/test_evaluation_v2.json` | `outputs/mobilenet_v2_staged_v2_retrained/test_predictions.csv` | `outputs/mobilenet_v2_staged_v2_retrained/test_plots_v2/` |
| Sensor-only InceptionTime | `outputs/sensor_only_inception_time_v2/test_evaluation.json` | `outputs/sensor_only_inception_time_v2/test_predictions.csv` | `outputs/sensor_only_inception_time_v2/test_plots/` |
| Multimodal fusion | `outputs/exp1_multimodal/test_evaluation_v2.json` | `outputs/exp1_multimodal/test_predictions.csv` | `outputs/exp1_multimodal/test_plots_v2/` |

Each prediction CSV includes the actual MMAct `sample_id`, ground-truth label, predicted label, confidence, and class probabilities.

### Training and Validation Artifacts

- Validation benchmark: `outputs/benchmark_validation/`
- Multimodal training history: `outputs/exp1_multimodal/history.json`
- Sensor-only training summary: `outputs/sensor_only_inception_time_v2/training_summary.json`
- Sensor-only runtime report: `outputs/sensor_only_inception_time_v2/runtime_report.json`

> [!NOTE]
> Preprocessing caches and temporary data are excluded from the artifact handover. Model checkpoint binaries remain subject to `.gitignore` rules and must be supplied separately or recreated through training if they are not present in a clone.

---

## 10. Reproducibility & CLI Execution Guide

### 10.1 Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

### 10.2 Execute Standalone Inference Demo
The standalone inference CLI ([`scripts/infer.py`](file:///d:/HAR_MMAct/scripts/infer.py)) supports three verified input modes:
1. **Multimodal Fused Inference:** Accepts both `--video` and wearable sensor CSVs (`--sensor_phone`, `--sensor_watch`, `--sensor_gyro`), passing both streams through the late fusion classification layer.
2. **Video Stream Alone:** Accepts `--video` alone, invoking `predict_video_only` via the auxiliary video classifier head of the multimodal architecture.
3. **Sensor Stream Alone:** Accepts sensor CSVs without video, invoking `predict_sensor_only` via the auxiliary sensor classifier head of the multimodal architecture.

#### Example Commands:

```powershell
# Mode 1: Multimodal fused inference using top model checkpoint
python scripts/infer.py `
    --video challenge-main/challenge-main/utils/sample_clip/carrying.mp4 `
    --sensor_phone path/to/acc_phone.csv `
    --sensor_watch path/to/acc_watch.csv `
    --sensor_gyro path/to/gyro.csv `
    --checkpoint outputs/exp1_multimodal/checkpoints/best_model.pth `
    --top_k 3

# Mode 2: Video stream alone
python scripts/infer.py `
    --video challenge-main/challenge-main/utils/sample_clip/carrying.mp4 `
    --checkpoint outputs/exp1_multimodal/checkpoints/best_model.pth

# Mode 3: Sensor stream alone
python scripts/infer.py `
    --sensor_phone path/to/acc_phone.csv `
    --sensor_watch path/to/acc_watch.csv `
    --sensor_gyro path/to/gyro.csv `
    --checkpoint outputs/exp1_multimodal/checkpoints/best_model.pth
```

#### Verified Demo Execution:
The test execution recorded during the audit ran:
```powershell
python scripts/infer.py `
    --video challenge-main/challenge-main/utils/sample_clip/carrying.mp4 `
    --checkpoint outputs/exp1_multimodal/checkpoints/best_model.pth
```
Output:
```text
============================================================
 HAR_MMAct: Inference Prediction
============================================================
[Mode: Video Stream Alone (Multimodal Architecture)]

Top-3 Activity Predictions:
  1. waving_hand     :  11.75%
  2. sitting         :  11.61%
  3. jumping         :  10.94%
============================================================
```

> [!NOTE]
> **Verification Scope:** This executed test confirms video-stream-only inference using the video branch and auxiliary classifier head of the multimodal architecture on an unlabelled sample video. It does not constitute proof of complete video-plus-sensor multimodal inference, which requires synchronized sensor time-series inputs.

### 10.3 Run Test Suite
```powershell
# Run accuracy and data-alignment regression tests
python -m pytest tests/test_accuracy_fixes.py -v

# Run model architecture and split isolation tests
python -m pytest tests/test_models.py tests/test_splits.py -v
```

---

## 11. Known Limitations & Recommendations

1. **Jumping recognition:** The multimodal test F1 for `jumping` is 0.2993, with 0.2619 recall. Rapid motion and transition phases remain difficult to distinguish.
2. **Running recall:** The multimodal model has 1.0000 precision but 0.4524 recall for `running`; it misses many true running samples.
3. **Class-dependent performance:** The three models show different strengths across activities. Accuracy alone is insufficient; Macro F1 and per-class metrics should also be considered.
4. **Video-only baseline:** Video-only MobileNetV2 achieved 42.49% test accuracy, substantially below sensor-only and multimodal results in this experiment.
5. **Dataset and split scope:** Results come from a custom subject-independent partition of MMAct. External datasets, new recording environments, and deployment conditions have not been evaluated here.
6. **Official blind test data:** The unlabelled official challenge test set cannot be used for local supervised evaluation. Reported test metrics refer only to the project's labelled held-out partition.

---

## 12. Conclusion

The HAR-MMAct project implements a software-only human activity recognition pipeline using genuine MMAct video and wearable inertial sensor data for 10 activity classes.

On the same 852 subject-independent held-out test samples, the multimodal MobileNetV2 + InceptionTime model achieved **76.76% accuracy** and **0.7545 Macro F1**, compared with 69.95% accuracy and 0.6665 Macro F1 for sensor-only InceptionTime, and 42.49% accuracy and 0.4317 Macro F1 for video-only MobileNetV2.

These results support multimodal fusion for this experimental setup. However, weak recognition of jumping and reduced running recall remain limitations. The evaluation artifacts and sample-level predictions are preserved to make the reported comparison auditable.

### Individual Training vs Held-Out Test Accuracy

These checkpoint-level plots compare the training accuracy recorded
during each selected checkpoint epoch with the corresponding held-out
test accuracy. Test data is evaluated after model selection, not once
per epoch.

- Video-only: `outputs/mobilenet_v2_staged_v2_retrained/plots/train_vs_test_accuracy_checkpoint.png`
- Sensor-only: `outputs/sensor_only_inception_time_v2/plots/train_vs_test_accuracy_checkpoint.png`
- Multimodal: `outputs/exp1_multimodal/plots/train_vs_test_accuracy_checkpoint.png`
- Combined comparison: `outputs/model_comparison/train_vs_test_accuracy_final_models.png`
- Source script: `scripts/plot_train_vs_test_accuracy.py`
- Data table: `outputs/model_comparison/train_vs_test_accuracy_checkpoint.csv`
