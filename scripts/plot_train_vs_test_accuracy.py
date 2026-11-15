import json
import csv
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "name": "Video-only MobileNetV2",
        "folder": "outputs/mobilenet_v2_staged_v2_retrained",
        "checkpoint": "outputs/mobilenet_v2_staged_v2_retrained/checkpoints/best_model_mobilenet_v2.pth",
        "history": "outputs/mobilenet_v2_staged_v2_retrained/history_mobilenet_v2.json",
        "evaluation": "outputs/mobilenet_v2_staged_v2_retrained/test_evaluation_v2.json",
        "metric_key": "video_only",
    },
    {
        "name": "Sensor-only InceptionTime",
        "folder": "outputs/sensor_only_inception_time_v2",
        "checkpoint": "outputs/sensor_only_inception_time_v2/checkpoints/best_model_cnn.pth",
        "history": "outputs/sensor_only_inception_time_v2/history_cnn.json",
        "evaluation": "outputs/sensor_only_inception_time_v2/test_evaluation.json",
        "metric_key": "sensor_only",
    },
    {
        "name": "Multimodal MobileNetV2 + InceptionTime",
        "folder": "outputs/exp1_multimodal",
        "checkpoint": "outputs/exp1_multimodal/checkpoints/best_model.pth",
        "history": "outputs/exp1_multimodal/history_mobilenet_v2.json",
        "evaluation": "outputs/exp1_multimodal/test_evaluation_v2.json",
        "metric_key": "multimodal",
    },
]

def read_sample_ids(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != 852 or len(set(ids)) != 852:
        raise ValueError(f"Invalid test sample IDs in {path}")
    return ids

results = []
reference_ids = None

for cfg in MODELS:
    checkpoint_path = ROOT / cfg["checkpoint"]
    history_path = ROOT / cfg["history"]
    evaluation_path = ROOT / cfg["evaluation"]

    for path in (checkpoint_path, history_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    epoch = int(checkpoint["epoch"])
    history = json.loads(history_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())

    assert evaluation["eval_split"] == "test"
    assert evaluation["split_strategy"] == "subject"
    assert evaluation["eval_samples_count"] == 852

    metrics = evaluation[cfg["metric_key"]]
    train_history = history["train_acc"]

    if epoch < 1 or epoch > len(train_history):
        raise ValueError(
            f"Checkpoint epoch {epoch} does not match history "
            f"length {len(train_history)}: {cfg['name']}"
        )

    # Training accuracy recorded in the checkpoint's own epoch.
    train_accuracy = float(train_history[epoch - 1])
    test_accuracy = float(metrics["accuracy"]) * 100.0

    if not (0 <= train_accuracy <= 100 and 0 <= test_accuracy <= 100):
        raise ValueError(f"Invalid accuracy values for {cfg['name']}")

    # Confirm all models use identical test sample identities.
    prediction_path = (
        ROOT / cfg["folder"] / "test_predictions.csv"
    )
    sample_ids = read_sample_ids(prediction_path)

    if reference_ids is None:
        reference_ids = sample_ids
    elif sample_ids != reference_ids:
        raise ValueError(
            f"Test sample identity/order mismatch: {cfg['name']}"
        )

    result = {
        "model": cfg["name"],
        "checkpoint_epoch": epoch,
        "training_accuracy_at_checkpoint_epoch_percent": round(train_accuracy, 4),
        "held_out_test_accuracy_percent": round(test_accuracy, 4),
        "test_samples": 852,
        "split": "subject-independent",
    }
    results.append(result)

    # Generate individual model graph.
    plot_dir = ROOT / cfg["folder"] / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        [f"Training\n(epoch {epoch})", "Held-out\ntest"],
        [train_accuracy, test_accuracy]
    )

    ax.set_title(
        f"Training vs Held-out Test Accuracy\n{cfg['name']}"
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, [train_accuracy, test_accuracy]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom"
        )

    fig.text(
        0.5, 0.01,
        "Training metric recorded during the selected checkpoint epoch; "
        "test evaluated once on 852 subject-independent samples.",
        ha="center",
        fontsize=8
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    output = plot_dir / "train_vs_test_accuracy_checkpoint.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    result["individual_plot"] = str(output.relative_to(ROOT))

# Combined final-model graph.
comparison_dir = ROOT / "outputs/model_comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 6))

labels = [r["model"] for r in results]
train_values = [
    r["training_accuracy_at_checkpoint_epoch_percent"]
    for r in results
]
test_values = [
    r["held_out_test_accuracy_percent"]
    for r in results
]

x = list(range(len(labels)))
width = 0.35

bars1 = ax.bar(
    [v - width / 2 for v in x],
    train_values,
    width,
    label="Training accuracy at checkpoint epoch"
)
bars2 = ax.bar(
    [v + width / 2 for v in x],
    test_values,
    width,
    label="Held-out test accuracy"
)

ax.set_title("Training vs Held-out Test Accuracy — Final Three Models")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=12, ha="right")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.3)

for bars in (bars1, bars2):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center",
            fontsize=9
        )

fig.tight_layout()
combined_plot = (
    comparison_dir / "train_vs_test_accuracy_final_models.png"
)
fig.savefig(combined_plot, dpi=200, bbox_inches="tight")
plt.close(fig)

# Save CSV and JSON.
csv_path = comparison_dir / "train_vs_test_accuracy_checkpoint.csv"
json_path = comparison_dir / "train_vs_test_accuracy_checkpoint.json"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "checkpoint_epoch",
            "training_accuracy_at_checkpoint_epoch_percent",
            "held_out_test_accuracy_percent",
            "test_samples",
            "split",
            "individual_plot",
        ],
        lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(results)

json_path.write_text(json.dumps({
    "method": (
        "Training accuracy recorded during the selected best-validation "
        "checkpoint epoch compared with held-out test accuracy from the "
        "same checkpoint. Test data was not evaluated epoch by epoch."
    ),
    "results": results,
}, indent=2) + "\n")

print("\nTRAINING VS TEST ACCURACY")
for r in results:
    print(
        f"{r['model']}: epoch {r['checkpoint_epoch']}, "
        f"train={r['training_accuracy_at_checkpoint_epoch_percent']:.2f}%, "
        f"test={r['held_out_test_accuracy_percent']:.2f}%"
    )

print("\nAll three sample-ID sequences match.")
print("Individual plots and combined plot generated.")
