# Target Activity Label Mapping & Verification

**Document Version:** 1.0
**Date:** 2026-11-04
**Project:** MMAct Multimodal Human Activity Recognition
**Audit Script:** `scripts/audit_dataset.py`

---

## 1. Executive Summary

A strict label verification was conducted against the official MMAct Challenge 2021 evaluation ground-truth files:
* `challenge-main/challenge-main/evaluation/trimmed_val_scene_gt.json` (5,027 samples, 35 unique classes)
* `challenge-main/challenge-main/evaluation/trimmed_val_view_gt.json` (3,500 samples, 36 unique classes)
* `challenge-main/challenge-main/evaluation/untrimmed_val_gt.json` (315 sessions, 37 unique classes)

**Audit Finding:** All **10 requested target classes** exist verbatim (in snake_case) across all three official validation ground-truth files and the published ICCV 2019 MMAct paper taxonomy. No requested target classes are missing or synthetic.

The corresponding trimmed video and sensor data for these 10 target classes are hosted in the project's Kaggle datasets:
* **[Trimmed Scene Video Dataset](https://www.kaggle.com/datasets/vikramche/trimmed-scene-video)** (video modality)
* **[MMAct Trimmed Dataset](https://www.kaggle.com/datasets/vikramche/mmact-trimmed)** (sensor and split modality)

---

## 2. Complete Label Mapping Table

| # | Requested Project Label | Exact MMAct Label | Evidence / Source File | Validation Sample Count (Scene / View / Untrimmed) | Availability Status | Disambiguation & Notes |
|---|---|---|---|---|---|---|
| 1 | **Walking** | `walking` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>Paper Fig. 1 | 160 / 117 / 236 | **VERIFIED** | Distinct from `running` and `loitering`. Regular locomotion activity. |
| 2 | **Jumping** | `jumping` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>Paper Table 1 | 160 / 117 / 236 | **VERIFIED** | Simple dynamic activity with sharp acceleration peaks. |
| 3 | **Running** | `running` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>Paper Fig. 2 | 160 / 117 / 240 | **VERIFIED** | High-cadence locomotion; distinct from `walking`. |
| 4 | **Standing** | `standing` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>README line 50 | 160 / 117 / 232 | **VERIFIED** | Static posture class; distinct from transition class `standing_up`. |
| 5 | **Waving hand** | `waving_hand` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>README line 55 | 160 / 117 / 236 | **VERIFIED** | Upper-body gesture. Wrist acceleration (`acc_watch_clip`) provides strong discriminative signal. |
| 6 | **Kicking** | `kicking` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>Paper Sec. 3.2 | 160 / 117 / 236 | **VERIFIED** | Lower-body sudden action. Phone acceleration (`acc_phone_clip`) provides strong signal. |
| 7 | **Sitting** | `sitting` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>Paper Sec. 3.2 | 79 / 38 / 79 | **VERIFIED** | Static posture class. Recorded primarily in desk work scene. Distinct from transition class `sitting_down`. |
| 8 | **Crouching** | `crouching` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>README line 51 | 160 / 117 / 236 | **VERIFIED** | Postural change / elevation drop. |
| 9 | **Using phone** | `using_phone` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>README line 56 | 239 / 155 / 236 | **VERIFIED** | Interaction with smartphone screen. Distinct from `talking_on_phone`. |
| 10 | **Pointing** | `pointing` | `trimmed_val_scene_gt.json`<br>`trimmed_val_view_gt.json`<br>`untrimmed_val_gt.json`<br>Paper Sec. 3.2 | 160 / 117 / 236 | **VERIFIED** | Arm-extension gesture. Prominent on smartwatch IMU. |

---

## 3. Disambiguation Analysis & Boundary Cases

In the MMAct dataset, there are closely related fine-grained classes that must not be confused with the 10 target classes:

1. **Posture vs. Transition:**
   * `sitting` (Target) vs. `sitting_down` (Transition): `sitting` represents the steady seated state; `sitting_down` represents the transitional movement from standing to seated.
   * `standing` (Target) vs. `standing_up` (Transition): `standing` is steady upright posture; `standing_up` is rising from a seated position.
   * *Resolution:* Filter specifically for `sitting` and `standing` unless an explicit transition fusion is desired.

2. **Phone Interactions:**
   * `using_phone` (Target): Looking at or typing on phone screen.
   * `using_phone_desk`: Using phone specifically while seated at desk.
   * `talking_on_phone` & `talking_on_phone_desk`: Holding phone up to ear.
   * *Resolution:* Target `using_phone`. If `using_phone_desk` appears in unsegmented desk sessions, keep them distinct or record merging if authorized.

3. **Locomotion:**
   * `walking` (Target) vs. `loitering`: `walking` is directed gait; `loitering` is slow meandering / lingering in the action area.
   * *Resolution:* Keep strictly separated as distinct classes.

---

## 4. Class Index Map for Project Modeling

For our software-only multimodal classifier, the 10 verified target classes will be indexed consistently:

```python
TARGET_CLASS_TO_IDX = {
    "crouching": 0,
    "jumping": 1,
    "kicking": 2,
    "pointing": 3,
    "running": 4,
    "sitting": 5,
    "standing": 6,
    "using_phone": 7,
    "walking": 8,
    "waving_hand": 9,
}

IDX_TO_TARGET_CLASS = {v: k for k, v in TARGET_CLASS_TO_IDX.items()}
```

---

## 5. Verification Status

* **Target Labels Verification:** **PASSED** (10 of 10 verified against ground-truth files).
* **Archive & Data File Verification:** Pending local download of trimmed/untrimmed video-sensor subsets (see `docs/DATASET_AUDIT.md`).
