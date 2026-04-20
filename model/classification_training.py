from pathlib import Path
import random
import json
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# =========================
# Configuration
# =========================
BASE_DIR = Path.home() / "Desktop"
CSV_PATH = BASE_DIR / "processed" / "buildings_all_with_crops.csv"
OUTPUT_DIR = BASE_DIR / "training_outputs" / "baseline_resnet50"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
IMAGE_SIZE = 128
USE_CLASS_WEIGHTS = True

LABEL_TO_IDX = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


# =========================
# Reproducibility
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducibility first
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================
# Dataset
# =========================
class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])  # shape: (H, W, 6), uint8
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # -> (6, H, W)

        y = LABEL_TO_IDX[row["damage_label"]]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# =========================
# Model
# =========================
class ResNet50SixChannel(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.backbone = resnet50(weights=weights)

        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight

        self.backbone.conv1 = new_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# =========================
# Metrics
# =========================
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    per_class_f1 = f1_score(all_targets, all_preds, average=None, labels=[0, 1, 2, 3])

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "targets": all_targets,
        "preds": all_preds,
    }


# =========================
# Main
# =========================
def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    required_splits = {"train", "test", "hold"}
    found_splits = set(df["split"].unique())
    missing = required_splits - found_splits
    if missing:
        raise ValueError(f"Missing required splits: {missing}")

    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    hold_df = df[df["split"] == "hold"].copy()

    print("\nSplit sizes:")
    print(df["split"].value_counts())

    print("\nTrain label distribution:")
    print(train_df["damage_label"].value_counts())

    train_dataset = XViewBuildingDataset(train_df)
    test_dataset = XViewBuildingDataset(test_df)
    hold_dataset = XViewBuildingDataset(hold_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    hold_loader = DataLoader(
        hold_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = ResNet50SixChannel(num_classes=4).to(device)

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2, 3]),
            y=train_df["damage_label"].map(LABEL_TO_IDX).values,
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\nUsing class weights:", class_weights.detach().cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_model_state = None
    best_test_macro_f1 = -1.0
    history = []

    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        test_metrics = evaluate(model, test_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_metrics["loss"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_f1_no_damage": float(test_metrics["per_class_f1"][0]),
            "test_f1_minor": float(test_metrics["per_class_f1"][1]),
            "test_f1_major": float(test_metrics["per_class_f1"][2]),
            "test_f1_destroyed": float(test_metrics["per_class_f1"][3]),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Macro F1: {test_metrics['macro_f1']:.4f}"
        )

        if test_metrics["macro_f1"] > best_test_macro_f1:
            best_test_macro_f1 = test_metrics["macro_f1"]
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is None:
        raise RuntimeError("Training failed: no best model state was saved.")

    best_model_path = OUTPUT_DIR / "best_model.pt"
    torch.save(best_model_state, best_model_path)
    print(f"\nBest model saved to: {best_model_path}")

    history_df = pd.DataFrame(history)
    history_path = OUTPUT_DIR / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

    model.load_state_dict(best_model_state)

    print("\nFinal evaluation on TEST split:")
    final_test = evaluate(model, test_loader, criterion, device)
    print(f"Macro F1: {final_test['macro_f1']:.4f}")
    print(
        classification_report(
            final_test["targets"],
            final_test["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
        )
    )

    print("\nFinal evaluation on HOLD split:")
    final_hold = evaluate(model, hold_loader, criterion, device)
    print(f"Macro F1: {final_hold['macro_f1']:.4f}")
    hold_report = classification_report(
        final_hold["targets"],
        final_hold["preds"],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
    )
    print(
        classification_report(
            final_hold["targets"],
            final_hold["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
        )
    )

    summary = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "best_test_macro_f1": float(best_test_macro_f1),
        "final_test_macro_f1": float(final_test["macro_f1"]),
        "final_hold_macro_f1": float(final_hold["macro_f1"]),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "hold_size": int(len(hold_df)),
    }

    summary_path = OUTPUT_DIR / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    hold_metrics_path = OUTPUT_DIR / "hold_classification_report.json"
    with open(hold_metrics_path, "w", encoding="utf-8") as f:
        json.dump(hold_report, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Hold report saved to: {hold_metrics_path}")


if __name__ == "__main__":
    main()