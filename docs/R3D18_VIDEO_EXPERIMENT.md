# R3D-18 video-only experiment

The dedicated Kaggle notebook `notebooks/kaggle_r3d18_experiment.ipynb` checks out `improved-models` and runs a separate experiment without changing existing benchmark artifact paths. The implementation uses torchvision's `R3D_18_Weights.KINETICS400_V1`, reverses the existing ImageNet normalization before applying the Kinetics-400 normalization, and raises a clear error if pretrained weight loading fails.

## Video-only dataset audit and split

The verified Kaggle dataset audit found **5,627 valid target-class MP4 videos**. This count includes clips without wearable sensor CSV files; the existing multimodal `MMActIndexer.index()` behavior is unchanged and continues to require `acc_phone`, `acc_watch`, and `gyro`. R3D-18 uses the separate `MMActIndexer.index_videos()` path, which filters to the ten verified target classes and does not scan or require sensor files.

The verified Kaggle audit counts for the existing subject-independent split are:

| Split | Subject IDs | Video samples |
|---|---|---:|
| Train | subject1–subject14 | 3,830 |
| Validation | subject15–subject17 | 945 |
| Held-out test | subject18–subject20 | 852 |

These are **verified Kaggle dataset audit counts**, not model results. The trainer enforces all three counts, exact subject groups, and 852 unique test sample IDs. It uses no official cross-scene or cross-view split files. It records split IDs and checks the test loader's IDs against the recorded held-out test IDs after selecting the best checkpoint by validation macro-F1. No test data is used for training or tuning.

## Run

Run from the repository root (Kaggle GPU with internet enabled):

```bash
python scripts/train_r3d18.py \
  --data_root data/genuine_mmact \
  --video_backbone r3d18 \
  --mode video_only \
  --experiment_name r3d18_kinetics400_video_only \
  --epochs 50 --batch_size 4 --lr 0.0001 --weight_decay 0.0001 \
  --dropout 0.35 --seed 42 \
  --output_dir outputs/video_model_improvements/r3d18 \
  --cache_dir outputs/cache --num_workers 2
```

The trainer checkpoints after every epoch and resumes automatically from `checkpoints/latest_model_r3d18.pth`. Validation macro-F1 selects the best checkpoint. Only after training is complete is that checkpoint evaluated on test. This code change does not start training.

Outputs: `checkpoints/best_model_r3d18.pth`, `checkpoints/latest_model_r3d18.pth`, `epoch_history.json` / `.csv`, `metrics.json`, `predictions.json`, and `experiment_manifest.json`.
