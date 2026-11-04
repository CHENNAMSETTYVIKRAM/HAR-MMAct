# MMAct Video Download & Minimal Subset Experiment Plan

**Date:** 2026-11-05  
**Workspace:** `D:\HAR_MMAct`  
**Status:** PROPOSED PLAN — PENDING USER APPROVAL  
**Target Activity Classes (10 Verified):** `crouching`, `jumping`, `kicking`, `pointing`, `running`, `sitting`, `standing`, `using_phone`, `walking`, `waving_hand`.

---

## 1. Executive Summary & Core Findings

1. **Box Repository Hierarchy Revealed by Screenshots:**
   - The official MMAct Challenge 2021 Box repository partitions video clips **by individual subject** (`subject01` to `subject20`), rather than requiring the download of monolithic 47 GB archives.
   - `trimmed/cross_view_video`: 20 subject archives, totaling **47.1 GB** (individual archives: 2.0 GB – 2.9 GB).
   - `trimmed/cross_scene_video`: 20 subject archives, totaling **41.4 GB** (individual archives: 1.8 GB – 2.5 GB).
   - `trimmed_sensor.zip`: Contains all 4 inertial sensor modalities (`acc_phone_clip`, `acc_watch_clip`, `gyro_clip`, `orientation_clip`) for all 20 subjects (estimated size: **~150–200 MB**).
   - `trimmed_splits.zip`: Contains official split configuration files (< 1 MB).

2. **Target Class Representation:**
   - Ground truth verification confirms that **19 out of 20 subjects** (including `subject16` and `subject8`) contain **10 out of 10 target activity classes**.

3. **Smallest Download Recommendation:**
   - **Minimalist Option (1 Subject Archive):** Download `trimmed_x-scene_subject16.zip` (**1.8 GB**) + `trimmed_sensor.zip` (~180 MB) = **~1.98 GB download total**.
     - Provides all 10 target classes.
     - Supports **Cross-View** (`cam1, 2, 4` vs `cam3`), **Cross-Scene** (`scene1, 2` vs `scene3, 4`), and **Session-Independent** splits without leakage.
     - *Limitation:* Does not support subject-independent evaluation (requires $\ge 2$ subjects).
   - **Rigorous Subject-Independent Option (2 Subject Archives):** Download `trimmed_x-scene_subject16.zip` (**1.8 GB**) + `trimmed_x-scene_subject8.zip` (**1.9 GB**) + `trimmed_sensor.zip` (~180 MB) = **~3.88 GB download total**.
     - Full subject-independent partitioning: Train on `subject16`, Evaluate on `subject8` (zero subject leakage).

---

## 2. Verified Dataset Evidence & Split Architecture

### 2.1 Ground Truth Annotations in Workspace
The challenge codebase (`challenge-main/challenge-main/evaluation/`) contains ground truth validation files:
- `trimmed_val_view_gt.json` (3,500 annotated clips): Evaluates view generalizability by holding out camera **`cam3`** across all 20 subjects and 4 scenes.
- `trimmed_val_scene_gt.json` (5,027 annotated clips): Evaluates environment generalizability by holding out scenes **`scene3` and `scene4`** across all 20 subjects and 4 cameras.

### 2.2 Target Class Availability per Subject
Cross-referencing the 10 target classes across validation ground truth files:
- `subject1` to `subject6`, `subject8` to `subject20`: **10 / 10 target classes present** (100% coverage).
- `subject7`: 9 / 10 target classes present (missing `sitting`).
- `subject16` (the smallest archive at 1.8 GB) has **10 / 10 verified target classes**.
- `subject8` (the second smallest archive at 1.9 GB) has **10 / 10 verified target classes**.

---

## 3. Box Archive Size Inventory (From Screenshots)

### A. `trimmed/cross_scene_video` (Folder ID: `134971395749`, Total: 41.4 GB)
| Archive Name | Compressed Size | Recommended Role |
|---|---|---|
| `trimmed_x-scene_subject16.zip` | **1.8 GB** | **Primary Subject (Train/Val)** |
| `trimmed_x-scene_subject8.zip` | **1.9 GB** | **Secondary Subject (Test)** |
| `trimmed_x-scene_subject9.zip` | 2.0 GB | Alternative |
| `trimmed_x-scene_subject10.zip` | 2.0 GB | Alternative |
| `trimmed_x-scene_subject14.zip` | 2.1 GB | Alternative |
| `trimmed_x-scene_subject20.zip` | 2.1 GB | Alternative |
| `trimmed_x-scene_subject11.zip` | 2.3 GB | Alternative |
| `trimmed_x-scene_subject15.zip` | 2.3 GB | Alternative |
| `trimmed_x-scene_subject17.zip` | 2.3 GB | Alternative |
| `trimmed_x-scene_subject12.zip` | 2.4 GB | Alternative |
| `trimmed_x-scene_subject19.zip` | 2.4 GB | Alternative |
| `trimmed_x-scene_subject18.zip` | 2.5 GB | Alternative |

### B. `trimmed/cross_view_video` (Folder ID: `134970947034`, Total: 47.1 GB)
| Archive Name | Compressed Size | Recommended Role |
|---|---|---|
| `trimmed_x-view_subject16.zip` | 2.0 GB | Alternative Primary |
| `trimmed_x-view_subject7.zip` | 2.1 GB | Incomplete (9/10 classes) |
| `trimmed_x-view_subject10.zip` | 2.2 GB | Alternative |
| `trimmed_x-view_subject9.zip` | 2.3 GB | Alternative |
| `trimmed_x-view_subject17.zip` | 2.4 GB | Alternative |
| `trimmed_x-view_subject14.zip` | 2.5 GB | Alternative |
| `trimmed_x-view_subject20.zip` | 2.5 GB | Alternative |
| `trimmed_x-view_subject11.zip` | 2.6 GB | Alternative |
| `trimmed_x-view_subject15.zip` | 2.6 GB | Alternative |
| `trimmed_x-view_subject13.zip` | 2.7 GB | Alternative |
| `trimmed_x-view_subject18.zip` | 2.7 GB | Alternative |
| `trimmed_x-view_subject19.zip` | 2.8 GB | Alternative |
| `trimmed_x-view_subject12.zip` | 2.9 GB | Alternative |

### C. Companion Modalities (Folder: `trimmed`, URL: `https://ent.box.com/s/5pg7zaw17r1em5czyqdokdrni92g6jte/`)
| Archive Name | Estimated Size | Contents |
|---|---|---|
| `trimmed_sensor.zip` | ~150 – 200 MB | Phone & watch accelerometer, gyroscope, orientation CSVs |
| `trimmed_splits.zip` | < 1 MB | Text split definitions |
| `trimmed_pose.zip` | ~300 – 500 MB | OpenPifPaf 2D skeleton keypoint JSONs (optional) |

---

## 4. Proposed Download Manifests

### Option 1: Minimum Viable Download (1 Subject)
*Use Case:* Fastest download, verifies multimodal fusion and cross-view or session-independent splits.

| # | Item | Box URL / Folder | Size | Classes | Split Usage |
|---|---|---|---|---|---|
| 1 | `trimmed_sensor.zip` | `trimmed/` | ~180 MB | 10/10 | All sensor streams |
| 2 | `trimmed_splits.zip` | `trimmed/` | < 1 MB | N/A | Split indices |
| 3 | `trimmed_x-scene_subject16.zip` | `trimmed/cross_scene_video/` | **1.8 GB** | **10/10** | Train / Val / Test (Cross-View or Session split) |
| **Total** | | | **~1.98 GB** | **10/10** | |

*Storage Breakdown:*
- Download Size: ~1.98 GB
- Extracted Video Size: ~2.1 GB
- Extracted Sensor Size: ~0.4 GB
- Working Disk Space Needed: **~4.5 GB** (reduces to ~2.5 GB after removing `.zip` files).

---

### Option 2: Rigorous Subject-Independent Download (2 Subjects — RECOMMENDED)
*Use Case:* Complete scientific rigor with zero subject leakage across train and test.

| # | Item | Box URL / Folder | Size | Classes | Split Usage |
|---|---|---|---|---|---|
| 1 | `trimmed_sensor.zip` | `trimmed/` | ~180 MB | 10/10 | All sensor streams |
| 2 | `trimmed_splits.zip` | `trimmed/` | < 1 MB | N/A | Split indices |
| 3 | `trimmed_x-scene_subject16.zip` | `trimmed/cross_scene_video/` | **1.8 GB** | **10/10** | **Training Set** |
| 4 | `trimmed_x-scene_subject8.zip` | `trimmed/cross_scene_video/` | **1.9 GB** | **10/10** | **Held-Out Validation & Test Set** |
| **Total** | | | **~3.88 GB** | **10/10** | |

*Storage Breakdown:*
- Download Size: ~3.88 GB
- Extracted Video Size: ~4.3 GB
- Extracted Sensor Size: ~0.4 GB
- Working Disk Space Needed: **~8.6 GB** (reduces to ~4.7 GB after removing `.zip` files).
- Current Free Space on `D:` drive: **238.96 GB** (ample headroom).

---

## 5. Box Capability & Selective Extraction Analysis

1. **Can individual `.mp4` video files be downloaded from Box without the `.zip` archive?**
   - **No.** In Box, files are packaged into `.zip` containers for each subject (`trimmed_x-scene_subject<ID>.zip`). Box does not provide a remote virtual file system to unpack `.zip` archives on the server before downloading.
   - However, Box **does** allow downloading individual subject `.zip` files directly (e.g., clicking download on `trimmed_x-scene_subject16.zip` alone, rather than downloading the entire 41.4 GB folder).
2. **Selective Extraction after Download:**
   - Once the ~1.8 GB zip file is on disk, we can selectively extract only the 10 target classes (`crouching`, `jumping`, `kicking`, `pointing`, `running`, `sitting`, `standing`, `using_phone`, `walking`, `waving_hand`) and discard the remaining 25+ non-target classes.
   - This reduces the extracted video footprint by ~65% (from ~2.1 GB down to ~750 MB per subject).

---

## 6. Experiment Compatibility Matrix

| Experiment Type | Option 1 (1 Subject) | Option 2 (2 Subjects) | Notes |
|---|---|---|---|
| **Sensor-Only Baseline** | YES | YES | Trained on InceptionTime 1D CNN |
| **Video-Only Baseline** | YES | YES | Trained on Pretrained ResNet-18 + Temporal Pooling |
| **Multimodal Fused HAR** | YES | YES | Features concatenated (256-d) $\rightarrow$ MLP classifier |
| **Cross-View Split** | YES (`cam1,2,4` vs `cam3`) | YES | Evaluates viewpoint generalizability |
| **Cross-Scene Split** | YES (`scene1,2` vs `scene3,4`)| YES | Evaluates background generalizability |
| **Subject-Independent Split** | NO (Single subject) | **YES (`subj16` vs `subj8`)** | Zero subject leakage |

---

## 7. Step-by-Step Manual Download Instructions

1. **Access the MMAct Box folder in your browser:**
   `https://ent.box.com/s/5pg7zaw17r1em5czyqdokdrni92g6jte/`
   *(Enter password if prompted)*
2. **Download Sensors & Splits:**
   - Click the download icon next to `trimmed_sensor.zip` (~180 MB).
   - Click the download icon next to `trimmed_splits.zip` (< 1 MB).
3. **Download Video Subject Archive(s):**
   - Click open `cross_scene_video` (Folder ID: `134971395749`).
   - Click download on `trimmed_x-scene_subject16.zip` (**1.8 GB**).
   - *(If Option 2 is approved)*: Click download on `trimmed_x-scene_subject8.zip` (**1.9 GB**).
4. **Place downloaded `.zip` files in:**
   `D:\HAR_MMAct\data\downloads\`
5. **Extraction command (ready to run once files are placed):**
   ```powershell
   & "C:\Users\chenn\AppData\Local\Programs\Python\Python311\python.exe" -c "
   import zipfile, os
   for z in ['trimmed_sensor.zip', 'trimmed_x-scene_subject16.zip']:
       path = os.path.join('data', 'downloads', z)
       if os.path.exists(path):
           print('Extracting', z)
           with zipfile.ZipFile(path) as f:
               f.extractall('data/genuine_mmact')
   "
   ```

---

## 8. Summary Recommendation for Approval

We recommend **Option 2 (Subject 16 + Subject 8 + Sensors, total 3.88 GB)**:
- Provides **10/10 target classes** in both subjects.
- Enables genuine **subject-independent training and evaluation** with mathematically clean split isolation.
- Extremely lightweight (3.88 GB download vs 88.5 GB full dataset).
- Runs within the existing project pipeline without requiring any architectural changes.
