# Initial Resource Audit: MMAct Human Activity Recognition

**Date:** 2026-11-04  
**Workspace:** `D:\HAR_MMAct`  
**Audit Type:** Read-only preliminary resource audit  

---

## 1. Workspace Tree

The current workspace contains the downloaded challenge codebase archive (unpacked into `challenge-main\challenge-main`), the original ICCV 2019 research paper, and a source link file:

```text
D:\HAR_MMAct\
├── Kong_MMAct_A_Large-Scale_Dataset_for_Cross_Modal_Human_Action_Understanding_ICCV_2019_paper.pdf
├── sources.txt
└── challenge-main\
    └── challenge-main\
        ├── index.html
        ├── LICENSE
        ├── README.md
        ├── assets\
        │   └── report\
        │       ├── DeepblueAI-Challenge report for MMAct Challenge.pdf
        │       ├── OPPO_mmact_challenge_report.pdf
        │       └── Raphael_unified_action_recognition_with_various_modalities.pdf
        ├── css\
        │   ├── animate.css
        │   ├── bootstrap.min.css
        │   ├── font-awesome.min.css
        │   ├── line-icons.css
        │   ├── magnific-popup.css
        │   ├── main.css
        │   ├── nivo-lightbox.css
        │   ├── owl.carousel.css
        │   ├── owl.theme.css
        │   ├── responsive.css
        │   └── slicknav.css
        ├── evaluation\
        │   ├── eval_mmact_trimmed.py
        │   ├── eval_mmact_untrimmed.py
        │   ├── trimmed_val_scene_gt.json
        │   ├── trimmed_val_scene_sample_format.json
        │   ├── trimmed_val_view_gt.json
        │   ├── trimmed_val_view_sample_format.json
        │   ├── untrimmed_val_gt.json
        │   └── untrimmed_val_sample_submission.json
        ├── fonts\
        │   └── [11 font assets: FontAwesome, Linearicons]
        ├── img\
        │   ├── bg1.png
        │   ├── dummy.png
        │   ├── email.png
        │   ├── hero-area.png
        │   ├── logo.png
        │   ├── logo.psd
        │   ├── mmactpaper.png
        │   └── team\
        │       └── [11 team avatar images]
        ├── js\
        │   └── [18 JavaScript vendor & script files]
        ├── php\
        │   └── form-process.php
        ├── scss\
        │   ├── main.scss
        │   └── responsive.scss
        └── utils\
            ├── pose_frame_extraction.py
            ├── sample_clip\
            │   └── carrying.mp4
            └── time_series_classifiers.py
```

---

## 2. Repository Summary

* **Repository Identity:** Official code and website repository for the **MMAct Challenge 2021** (hosted at the CVPR 2021 ActivityNet Workshop).
* **License:** Creative Commons Attribution-NonCommercial 4.0 International Public License (`CC BY-NC 4.0`), located at `challenge-main\challenge-main\LICENSE`. Commercial usage is restricted without prior authorization from the licensors (Hitachi, Ltd. R&D Group).
* **Architecture & Components:**
  1. **Challenge Website (`index.html`):** Static web frontend providing challenge descriptions, organizer information, timelines, and competition download forms/server links.
  2. **Evaluation Framework (`evaluation/`):** Standalone Python benchmarking tools evaluating:
     - Trimmed Video Action Recognition (mAP classification metric via `eval_mmact_trimmed.py`).
     - Untrimmed Video Temporal Action Localization (Interpolated AP across IoU thresholds 0.50–0.95 via `eval_mmact_untrimmed.py`).
     - Ground-truth files and sample prediction formats for validation splits (`trimmed_val_scene_gt.json`, `trimmed_val_view_gt.json`, `untrimmed_val_gt.json`).
  3. **Sensor Toolkit (`utils/time_series_classifiers.py`):** Baseline preprocessing routines for multi-modal time-series sensor alignment (`datetime_merging`), sliding-window generation (`create_windowed_data`), and time-series deep learning classification utilizing `tsai` / `fastai` (InceptionTime).
  4. **Keypoint Extraction Helper (`utils/pose_frame_extraction.py`):** Placeholder script for frame iteration using OpenCV to align with OpenPifPaf pose keypoints.
  5. **Participant Reports (`assets/report/`):** Post-competition workshop technical reports from top participants (DeepBlueAI, OPPO Research Institute, University of Koblenz).

---

## 3. Expected Dataset Layout

Based on `README.md`, `time_series_classifiers.py`, and the ground-truth JSON files, the dataset is structured into **trimmed** and **untrimmed** branches:

```text
MMAct_Dataset/
├── trimmed/
│   ├── video/ (or split folders: cross_scene/, cross_view/)
│   │   └── <split>/
│   │       └── trainval/
│   │           └── subject<1..20>/
│   │               └── cam<1..4>/
│   │                   └── scene<1..4>/
│   │                       └── session<1..5>/
│   │                           └── <action_label>.mp4
│   ├── sensors/
│   │   ├── acc_phone_clip/
│   │   ├── acc_watch_clip/
│   │   ├── gyro_clip/
│   │   └── orientation_clip/
│   │       └── subject<1..20>/
│   │           └── scene<1..4>/
│   │               └── session<1..5>/
│   │                   └── <action_label>.csv
│   ├── pose/
│   │   └── [OpenPifPaf JSON keypoints per frame / sequence]
│   └── splits/
│       ├── [cross-scene split definitions]
│       └── [cross-view split definitions]
│
└── untrimmed/
    ├── video/
    │   └── trainval/
    │       └── cam<1..4>/
    │           └── subject<1..20>/
    │               └── scene<1..4>/
    │                   └── session<1..5>/
    │                       └── <YYYYMMDD-HHMMSS>.mp4 (filename represents end timestamp)
    ├── sensor/ (or sensors/)
    │   ├── acc_phone_clip/
    │   ├── acc_watch_clip/
    │   ├── gyro_clip/
    │   └── orientation_clip/
    │       └── subject<1..20>/
    │           └── scene<1..4>/
    │               └── session<1..5>/
    │                   └── <filename>.csv
    ├── annotation/ (or sessions/)
    │   └── trainval/
    │       └── subject<1..20>/
    │           └── scene<1..4>/
    │               └── session<1..5>/
    │                   └── subject<S>_scene<C>_session<E>.txt
    └── splits/
        └── [cross-session split definitions]
```

### Key Dataset Specifics:
1. **Camera Views:** 4 static high-resolution surveillance cameras (`cam1` through `cam4` at 1920×1080, 30 fps) mounted at room corners, plus 1 egocentric smart glass view (`ego`, 1280×720, 30 fps).
2. **Sensor Modalities:**
   - `acc_phone_clip`: 3-axis phone acceleration in trousers pocket (100 Hz, m/s²).
   - `acc_watch_clip`: 3-axis smartwatch acceleration on right wrist (100 Hz, m/s²).
   - `gyro_clip`: 3-axis phone gyroscope in trousers pocket (50 Hz).
   - `orientation_clip`: 3-axis phone orientation (`azimuth`, `pitch`, `roll`) (50 Hz).
   - *Note on sensors:* Body sensors are invariant to camera perspective, so sensor folders contain no camera subdirectories.
3. **Annotations:**
   - **Trimmed data:** Annotation is implicit in the file name itself (e.g., `drinking.mp4`, `drinking.csv`).
   - **Untrimmed data:** Tabular text files mapping time spans to actions:
     `YYYY/MM/DD HH:MM:SS.fff-YYYY/MM/DD HH:MM:SS.fff-<action_name>`

---

## 4. Source-Link Inventory & Classification

### A. Links Saved in `sources.txt`
The file `D:\HAR_MMAct\sources.txt` contains:
1. `https://github.com/mmact19/challenge?utm_source=chatgpt.com` — **Official MMAct documentation/repository** (MMAct challenge repository mirror with tracking parameter)
2. `https://github.com/mmact19/challenge` — **Official MMAct documentation/repository** (Canonical official challenge repository)

*Note:* `sources.txt` does not directly list individual Box archive URLs.

### B. Comprehensive Inventory of Official Sources Found in Repository Assets
Inspecting `challenge-main\challenge-main\index.html`, `README.md`, and paper:

| URL | Classification | Purpose / Content Description |
|---|---|---|
| `https://github.com/mmact19/challenge` | **Official MMAct documentation/repository** | Challenge codebase, evaluation tools, and specifications |
| `https://mmact19.github.io/challenge/` | **Official MMAct documentation/repository** | Official challenge landing page hosted on GitHub Pages |
| `https://mmact19.github.io/2019/` | **Official MMAct documentation/repository** | Original ICCV 2019 dataset project home page |
| `https://docs.google.com/forms/d/e/1FAIpQLSdg-6E1ogV5VNA9GxhpAG8ehlBWtEH-UTgPPxYWe6d9fLdfqw/viewform` | **Dataset Access Request** | Google Form required to obtain download permissions and Box credentials for Train/Val archives |
| `https://ent.box.com/s/gz113lr0m33gmkvb9vuw6cs1k9s71nhc` | **Test Data Archive (Box)** | Official Box cloud folder hosting challenge test data (password-protected) |
| `https://competitions.codalab.org/competitions/31438` | **Leaderboard / Benchmark** | CodaLab evaluation server for Task 1 (Trimmed Action Recognition) |
| `https://competitions.codalab.org/competitions/31385` | **Leaderboard / Benchmark** | CodaLab evaluation server for Task 2 (Untrimmed Temporal Action Localization) |
| `http://activity-net.org/challenges/2021/index.html` | **Other** | Workshop page for CVPR 2021 ActivityNet Challenge |
| `https://github.com/vita-epfl/openpifpaf` | **Other** | Tool used by authors for pose extraction |
| `https://github.com/hfawaz/InceptionTime` | **Other** | Upstream reference library for sensor time-series classification |

---

## 5. Necessary vs. Optional Dataset Components

For the software-only **paired video-sensor human activity recognition (classification)** project:

### Necessary Components:
1. **Trimmed Video Clips (`trimmed/video/`):**
   - RGB video files corresponding to selected evaluation splits (cross-scene or cross-view).
   - Serves as the primary visual stream.
2. **Trimmed Inertial Sensor Streams (`trimmed/sensors/`):**
   - Specifically `acc_phone_clip` (phone acceleration, 100 Hz) and `acc_watch_clip` (wrist acceleration, 100 Hz).
   - `gyro_clip` (gyroscope, 50 Hz): standard complementary inertial modality for 6-DoF body dynamics.
3. **File Labels / Split Ground Truth:**
   - In trimmed data, labels are embedded directly in the filename (`<label>.csv`, `<label>.mp4`).
   - Official validation ground truth JSON files (`trimmed_val_scene_gt.json` or `trimmed_val_view_gt.json`) for validation benchmarks.

### Optional Components:
1. **`orientation_clip`:**
   - 3-axis orientation signals (`azimuth`, `pitch`, `roll`). Often redundant when accelerometer and gyroscope are combined, but supported by `time_series_classifiers.py`.
2. **Pose Keypoints (`trimmed/pose/`):**
   - OpenPifPaf 2D skeleton joints. Optional intermediate modality; not strictly required if learning directly from video frames or if skeleton models are not part of the baseline.
3. **Untrimmed Video & Sensors (`untrimmed/`):**
   - Contains 3–8 minute continuous sessions and temporal boundary files (`untrimmed/annotation/`). Only necessary if tackling Task 2 (temporal action localization) or unsegmented stream parsing.
4. **Passive Modalities (Wi-Fi CSI, Pressure Signals):**
   - Mentioned in the original ICCV 2019 paper (7 total modalities recorded), but not utilized in the challenge evaluation scripts or sensor baseline scripts.
5. **Egocentric Camera Stream (`cam_ego` / Google Glass):**
   - 720p wearable camera view; optional unless specifically benchmarking first-person vs. third-person perception.

---

## 6. Supported Data Formats, Split Files, and Annotations

### Data Format Support:
* **Trimmed Data Support:** Fully supported.
  - Evaluation tool: `evaluation/eval_mmact_trimmed.py`
  - Baseline pipeline: `time_series_classifiers.py::trimmed_data_example`
  - Protocols supported: **Cross-Scene** (Train: scene1, scene2, scene3; Val: scene4) and **Cross-View** (Train: cam1, cam2, cam3; Val: cam4).
* **Untrimmed Data Support:** Fully supported.
  - Evaluation tool: `evaluation/eval_mmact_untrimmed.py`
  - Baseline pipeline: `time_series_classifiers.py::untrimmed_data_example`
  - Protocols supported: **Cross-Session** (Train: session1, session2, session3; Val: session4).

### Split Files & Ground Truth Formats:
1. **Trimmed Action Recognition Ground Truth (`evaluation/trimmed_val_scene_gt.json`, `trimmed_val_view_gt.json`):**
   ```json
   {
     "annotations": {
       "cross_scene/trainval/subject1/cam1/scene3/session4/drinking.mp4": [
         {
           "label": "drinking"
         }
       ]
     }
   }
   ```
2. **Trimmed Submission / Prediction Format:**
   ```json
   {
     "results": {
       "cross_scene/trainval/subject1/cam1/scene3/session4/drinking.mp4": [
         {
           "label": "drinking",
           "score": 0.95
         }
       ]
     }
   }
   ```
3. **Untrimmed Temporal Action Localization Ground Truth (`evaluation/untrimmed_val_gt.json`):**
   ```json
   {
     "annotations": {
       "cam1/scene1/subject1/session3/20181016-140459.mp4": [
         {
           "label": "standing",
           "segment": [62.03, 66.32]
         },
         {
           "label": "crouching",
           "segment": [70.58, 75.12]
         }
       ]
     }
   }
   ```
4. **Untrimmed Raw Annotation Files (`untrimmed/annotation/`):**
   Text file with format: `<start_time>-<end_time>-<action_name>` with timestamps formatted as `YYYY/MM/DD HH:MM:SS.fff`.

---

## 7. Size & Storage Estimates

*(All values below are marked as **estimates** based on paper specifications, repo docs, and user notes; unverified until actual Box archive inspection)*

| Component | Estimated Size (Download Archive) | Estimated Size (Extracted) | Basis of Estimate |
|---|---|---|---|
| **Trimmed Video Archives (Full)** | ~77 GB – 92 GB *(Estimate)* | ~100 GB – 130 GB *(Estimate)* | User note (~77–92 GB archive); 36,764 clips × ~4s @ 1080p H.264 |
| **Trimmed Subset (Cross-Scene or Cross-View)** | ~25 GB – 35 GB *(Estimate)* | ~35 GB – 50 GB *(Estimate)* | Sub-partitioning by camera or scene protocol |
| **Trimmed Sensor Data (`.csv`)** | ~1 GB – 3 GB *(Estimate)* | ~2 GB – 6 GB *(Estimate)* | 100 Hz 3-axis CSV data for 36k short clips |
| **Trimmed Pose Keypoints (`.json`)** | ~3 GB – 6 GB *(Estimate)* | ~5 GB – 10 GB *(Estimate)* | 2D OpenPifPaf coordinates per frame |
| **Untrimmed Video Archives (Full)** | ~150 GB – 250 GB+ *(Estimate)* | ~250 GB – 400 GB+ *(Estimate)* | 1,900+ sessions × 3–8 min @ 1080p 30 fps across 4-5 cameras |
| **Untrimmed Sensor Data (`.csv`)** | ~2 GB – 5 GB *(Estimate)* | ~4 GB – 10 GB *(Estimate)* | Continuous raw sensor logs at 100 Hz |

---

## 8. Target Class Verification Status

> [!IMPORTANT]
> **Label Verification Pending:** Do not assume a predefined 10-class subset exists in the base dataset.

* **Audit Finding:**
  - The official MMAct dataset as detailed in the ICCV 2019 publication comprises **37 action classes** (16 complex actions, 12 simple actions, 9 desk work actions).
  - The ground truth validation files in the repository inspect as follows:
    * `trimmed_val_scene_gt.json`: **35 distinct classes**
    * `trimmed_val_view_gt.json`: **36 distinct labels** (35 canonical classes + 1 duplicate due to casing: `Carrying_light` vs `carrying_light`)
    * `untrimmed_val_gt.json`: **37 distinct classes** (includes `talking_on_phone_desk` and `using_phone_desk`)
  - **No 10-class split is defined or referenced in the repository.** If the project requires a 10-class subset, that mapping must be formally specified by project requirements or derived from a custom task configuration once archive annotation files are inspected.

---

## 9. Unresolved Questions & Blockers

1. **Direct Box URLs & Access Credentials:**
   - `sources.txt` currently only contains the GitHub repository URL (`https://github.com/mmact19/challenge`).
   - The official download mechanism requires filling a Google Form to receive the Box folder URL and access password for the train/val archives. The test set Box link (`https://ent.box.com/s/gz113lr0m33gmkvb9vuw6cs1k9s71nhc`) is password-protected.
   - *Status:* Blocked until direct Box links and/or credentials are confirmed.
2. **10-Class Target Subset Definition:**
   - The target 10 classes referenced in the project objectives have not yet been identified or mapped against the 35–37 dataset classes.
   - *Status:* Pending user clarification or project specification document.
3. **Evaluation Protocol Selection:**
   - The project must select between **Cross-Scene** (train on scenes 1–3, evaluate on scene 4 occlusion) and **Cross-View** (train on cams 1–3, evaluate on cam 4).
   - This determines which minimal archive slices need to be downloaded to avoid pulling the full ~77–92 GB dataset.
4. **Environment Dependencies:**
   - Preprocessing and baseline code in `time_series_classifiers.py` depends on `tsai`, `fastai`, `pytz`, and `torch`, which are not currently installed in the global Python environment. A dedicated virtual environment will be required prior to running code.
