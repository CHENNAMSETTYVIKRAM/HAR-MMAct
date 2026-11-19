# HAR-MMAct — Fast Accuracy Improvement Report

## Scope and safeguards

This report covers only the existing Custom CNN and MobileNetV2 video branches. No multimodal or sensor-only implementation behavior was altered for a model change, and no held-out test artifact was used to select or tune a configuration. Existing benchmark checkpoints, histories, metrics, and plots were not replaced.

## Confirmed implementation problems

1. **Custom CNN could not honor a less-aggressive `--dropout` setting.**  
   `MultimodalHARModel` used `dropout=max(0.35, dropout)` for the Custom CNN, overriding values such as `0.20` or `0.25`. This was fixed by passing the configured dropout directly.

2. **Custom CNN V2 spatial-dropout defaults were comparatively aggressive.**  
   The branch used `Dropout2d(0.15)` and `Dropout2d(0.20)` unconditionally. The branch now exposes separate spatial-dropout arguments with less aggressive defaults (`0.10` and `0.15`) while preserving the existing architecture and checkpoint parameter layout.

3. **Video-only training included unused modalities in the optimizer.**  
   Video-only runs instantiated the complete fusion model and previously allowed all trainable model parameters, including sensor/fusion parameters, to enter the optimizer parameter list. The training setup now limits AdamW to the active video branch and video-only classifier for `custom_cnn`, `original_cnn`, `cnn`, `improved_cnn`, `mobilenet_v2`, and `mobilenet` video-only runs.

4. **MobileNetV2 initialization and staged state were not sufficiently auditable.**  
   The branch now records the initialization source and ImageNet normalization constants. Staged-finetuning manifests will record pretrained/freeze/staged/unfreeze fields. The staged optimizer now separates the MobileNet backbone and the MobileNet temporal/projection plus video-only head, avoiding unused modalities in the head group.

5. **Original/retrained split difference is a dataset membership difference, not proven leakage.**  
   Retrained manifests add exactly four samples to training and change standing from 355 to 359. Validation and test counts and distributions remain unchanged. Every recorded isolation check is passed. The underlying historical sample identities are not persisted sufficiently to independently prove whether those four records are newly indexed, but the manifests establish no split-overlap regression.

## Fixes applied

- `src/models/fusion_model.py`: removed the Custom CNN minimum-dropout override.
- `src/models/video_branch.py`: made Custom CNN spatial-dropout configurable; added MobileNet initialization-source and normalization audit attributes.
- `scripts/train.py`: added active video-only parameter selection, manifest/model audit fields, and MobileNet-specific staged head/backbone grouping.
- `tests/test_fast_accuracy_diagnostics.py`: added shape, gradient, small-subset, MobileNet audit, label mapping, and split-difference regression coverage.
- `docs/FAST_ACCURACY_IMPROVEMENT_REPORT.md`: this report.

Existing source backup directories and output benchmark directories were left untouched.

## Custom CNN diagnostic specification

The added diagnostics verify:

- branch output `(B,128)`;
- video-only logits `(B,10)`;
- finite outputs;
- trainable parameter presence;
- non-zero convolution and video-head gradients;
- small balanced subset behavior on synthetic data.

The implementation cannot claim an observed overfit result until the test suite is executed in a Python environment with the repository dependencies. No held-out test data is involved.

## MobileNetV2 initialization verification

The implementation is designed to expose:

- `torchvision-default` for official default weights;
- `torchvision-legacy` for legacy pretrained loading fallback;
- `torchvision-random` if explicitly unweighted;
- `pure-pytorch-random` or `pure-pytorch-external` fallback state.

The source now also records ImageNet normalization mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. Runtime loading success is network/environment dependent and should be reported by executing the verification test.

## Tests added

`tests/test_fast_accuracy_diagnostics.py` includes:

- canonical label mapping round trip;
- Custom CNN forward shape and gradient coverage;
- 10-sample synthetic overfit diagnostic;
- MobileNet normalization, frozen backbone, trainable temporal projection, and embedding shape;
- auditable MobileNet weight source;
- exact four-sample/retrain standing-count investigation.

## Tests passed/failed

No pytest process could be launched by the repository-editing tools available in this task. Therefore no test pass/fail result is fabricated. The exact command to run locally is:

```bash
python -m pytest -q tests/test_fast_accuracy_diagnostics.py tests/test_models.py tests/test_accuracy_fixes.py
```

Relevant existing suite without the sensor-specific file:

```bash
python -m pytest -q tests/test_models.py tests/test_accuracy_fixes.py tests/test_label_mapping.py tests/test_splits.py tests/test_fast_accuracy_diagnostics.py
```

## Files changed

- `src/models/fusion_model.py`
- `src/models/video_branch.py`
- `scripts/train.py`
- `tests/test_fast_accuracy_diagnostics.py`
- `docs/FAST_ACCURACY_IMPROVEMENT_REPORT.md`

## Exact commands for next short experiments

Run diagnostics only; do not use test metrics:

```bash
python -m pytest -q tests/test_fast_accuracy_diagnostics.py
python scripts/train.py --smoke_test --mode video_only --video_backbone custom_cnn --epochs 10 --batch_size 8 --lr 1e-3 --dropout 0.20 --no_class_weights --benchmark_warmup --output_dir outputs/diagnostics/cnn_dropout020
python scripts/train.py --smoke_test --mode video_only --video_backbone custom_cnn --epochs 10 --batch_size 8 --lr 1e-3 --dropout 0.25 --no_class_weights --benchmark_warmup --output_dir outputs/diagnostics/cnn_dropout025
python scripts/train.py --smoke_test --mode video_only --video_backbone mobilenet_v2 --epochs 10 --batch_size 8 --lr 1e-4 --dropout 0.20 --freeze_video --benchmark_warmup --output_dir outputs/diagnostics/mobilenet_frozen10
python scripts/train.py --smoke_test --mode video_only --video_backbone mobilenet_v2 --epochs 15 --batch_size 8 --lr 2e-4 --dropout 0.20 --staged_finetune --unfreeze_epoch 15 --unfreeze_lr 1e-5 --benchmark_warmup --output_dir outputs/diagnostics/mobilenet_staged15
```

After the synthetic commands pass, use short real validation-only screenings (for example, 10–15 epochs) with the existing subject split and never evaluate or tune against the test set.

## Preserved benchmark assets

No files under `outputs/` were overwritten or deleted. The original and retrained checkpoints, histories, manifests, summaries, plots, and benchmark JSON/CSV artifacts remain unchanged.
