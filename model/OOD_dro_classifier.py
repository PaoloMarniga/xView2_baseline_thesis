from pathlib import Path
import random
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# =========================
# Configuration
# =========================
BASE_DIR = Path.home() / "Desktop"

CSV_PATH = BASE_DIR / "OOD_processed" / "buildings_all_OOD_with_crops.csv"

OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "baseline_resnet50_dro"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

TRAIN_SPLIT = "OOD_train"
TEST_SPLIT = "OOD_test"
HOLD_SPLIT = "OOD_hold"

USE_CLASS_WEIGHTS = False

UNKNOWN_GROUP = "__unknown__"

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
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# Device
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================
# Dataset with GROUP
# =========================
class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe, group_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.group_to_idx = group_to_idx
        self.unknown_group_id = group_to_idx[UNKNOWN_GROUP]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        y = LABEL_TO_IDX[row["damage_label"]]

        group = row["disaster"]

        # Training groups are known.
        # Test or hold groups unseen during training are mapped to "__unknown__".
        group_id = self.group_to_idx.get(group, self.unknown_group_id)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(group_id, dtype=torch.long),
        )


# =========================
# Model
# =========================
class ResNet50SixChannel(nn.Module):
    def __init__(self, num_classes=4):
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
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight

        self.backbone.conv1 = new_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# =========================
# Evaluation
# =========================
def evaluate(model, loader, device):
    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y.cpu().numpy())

    macro_f1 = f1_score(targets_all, preds_all, average="macro")

    return {
        "macro_f1": macro_f1,
        "preds": preds_all,
        "targets": targets_all,
    }


# =========================
# MAIN
# =========================
def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    train_df = df[df["split"] == TRAIN_SPLIT].copy()
    test_df = df[df["split"] == TEST_SPLIT].copy()
    hold_df = df[df["split"] == HOLD_SPLIT].copy()

    # =========================
    # Group mapping
    # =========================
    # Important:
    # The DRO group mapping is built only from training groups.
    # Test and hold disasters that were not seen during training are mapped
    # to "__unknown__" during evaluation.
    unique_train_groups = sorted(train_df["disaster"].unique())

    group_to_idx = {g: i for i, g in enumerate(unique_train_groups)}
    group_to_idx[UNKNOWN_GROUP] = len(group_to_idx)

    print("\nNumber of training groups:", len(unique_train_groups))
    print("Group mapping:")
    for group_name, group_id in group_to_idx.items():
        print(f"  {group_id}: {group_name}")

    unseen_test_groups = sorted(set(test_df["disaster"].unique()) - set(unique_train_groups))
    unseen_hold_groups = sorted(set(hold_df["disaster"].unique()) - set(unique_train_groups))

    print("\nUnseen TEST groups mapped to __unknown__:", unseen_test_groups)
    print("Unseen HOLD groups mapped to __unknown__:", unseen_hold_groups)

    train_loader = DataLoader(
        XViewBuildingDataset(train_df, group_to_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        XViewBuildingDataset(test_df, group_to_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    hold_loader = DataLoader(
        XViewBuildingDataset(hold_df, group_to_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = get_device()
    print("\nUsing device:", device)

    model = ResNet50SixChannel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1
    best_state = None

    print("\nStarting DRO-lite training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for x, y, g in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            g = g.to(device)

            optimizer.zero_grad()

            logits = model(x)

            # Per-sample loss
            losses = F.cross_entropy(logits, y, reduction="none")

            # DRO-lite over groups present in the current training batch
            group_losses = []
            unique_g = g.unique()

            for group_id in unique_g:
                mask = g == group_id
                group_loss = losses[mask].mean()
                group_losses.append(group_loss)

            group_losses = torch.stack(group_losses)

            # Higher-loss groups receive higher weight
            weights = group_losses / (group_losses.sum() + 1e-8)

            loss = (weights * group_losses).sum()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch + 1} Loss: {total_loss:.4f}")

        test_metrics = evaluate(model, test_loader, device)

        print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")

        if test_metrics["macro_f1"] > best_f1:
            best_f1 = test_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    # =========================
    # Final evaluation
    # =========================
    if best_state is not None:
        model.load_state_dict(best_state)

    final_test = evaluate(model, test_loader, device)
    final_hold = evaluate(model, hold_loader, device)

    print("\nFinal TEST Macro F1:", final_test["macro_f1"])
    print("\nFinal HOLD Macro F1:", final_hold["macro_f1"])

    print("\nTEST REPORT:")
    print(
        classification_report(
            final_test["targets"],
            final_test["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(len(IDX_TO_LABEL))],
        )
    )

    print("\nHOLD REPORT:")
    print(
        classification_report(
            final_hold["targets"],
            final_hold["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(len(IDX_TO_LABEL))],
        )
    )


if __name__ == "__main__":
    main()