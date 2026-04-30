##############
# OOD_beta_tcvae_classifier_fixed.py
##############

from pathlib import Path
import random
import json
import copy
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


##############
# 1. Configuration
##############

BASE_DIR = Path.home() / "Desktop"

CSV_PATH = BASE_DIR / "OOD_processed" / "buildings_all_OOD_with_crops.csv"
ENCODER_PATH = BASE_DIR / "OOD_training_outputs" / "beta_tcvae_fixed" / "beta_tcvae_encoder.pt"

OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "beta_tcvae_classifier_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 2

CLASSIFIER_LR = 1e-4
ENCODER_LR = 1e-5
WEIGHT_DECAY = 1e-4

USE_CLASS_WEIGHTS = True
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
USE_SQRT_CLASS_WEIGHTS = False

FREEZE_ENCODER = False
USE_MU_ONLY = True

LATENT_DIM = 128

TRAIN_SPLIT = "OOD_train"
TEST_SPLIT = "OOD_test"
HOLD_SPLIT = "OOD_hold"

LABEL_TO_IDX = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


##############
# 2. Reproducibility and device
##############

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


##############
# 3. Dataset
##############

class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe, train=False):
        self.df = dataframe.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        if self.train:
            if random.random() < 0.5:
                x = np.flip(x, axis=2).copy()

            if random.random() < 0.5:
                x = np.flip(x, axis=1).copy()

            if random.random() < 0.25:
                noise = np.random.normal(0.0, 0.01, size=x.shape).astype(np.float32)
                x = np.clip(x + noise, 0.0, 1.0)

        y = LABEL_TO_IDX[row["damage_label"]]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


##############
# 4. Encoder and classifier
##############

class BetaTCVAEEncoder(nn.Module):
    def __init__(self, latent_dim=128, use_mu_only=True):
        super().__init__()

        self.use_mu_only = use_mu_only

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Flatten(),
        )

        self.flatten_dim = 256 * 8 * 8

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if self.use_mu_only:
            return mu

        return torch.cat([mu, logvar], dim=1)


class LatentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, z):
        return self.classifier(z)


class BetaTCVAEClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits


##############
# 5. Loss
##############

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_unweighted = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(-ce_unweighted)
        focal_factor = (1.0 - pt) ** self.gamma

        loss = focal_factor * ce_unweighted

        if self.weight is not None:
            loss = self.weight[targets] * loss

        return loss.mean()


##############
# 6. Evaluation
##############

def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()

    total_loss = 0.0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False)

        for x, y in progress:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    macro_f1 = f1_score(
        targets_all,
        preds_all,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    per_class_f1 = f1_score(
        targets_all,
        preds_all,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    pred_counts = pd.Series(preds_all).value_counts().sort_index().to_dict()
    target_counts = pd.Series(targets_all).value_counts().sort_index().to_dict()

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "preds": preds_all,
        "targets": targets_all,
        "pred_counts": pred_counts,
        "target_counts": target_counts,
    }


def print_prediction_distribution(metrics, title):
    print(f"\n{title} prediction distribution:")
    for idx in range(4):
        label = IDX_TO_LABEL[idx]
        count = metrics["pred_counts"].get(idx, 0)
        print(f"{label}: {count}")

    print(f"\n{title} true distribution:")
    for idx in range(4):
        label = IDX_TO_LABEL[idx]
        count = metrics["target_counts"].get(idx, 0)
        print(f"{label}: {count}")


##############
# 7. Main
##############

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    print("\nAvailable splits:")
    print(df["split"].value_counts())

    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    train_df = df[df["split"] == TRAIN_SPLIT].copy()
    test_df = df[df["split"] == TEST_SPLIT].copy()
    hold_df = df[df["split"] == HOLD_SPLIT].copy()

    if train_df.empty:
        raise ValueError(f"No rows found for split: {TRAIN_SPLIT}")

    if test_df.empty:
        raise ValueError(f"No rows found for split: {TEST_SPLIT}")

    if hold_df.empty:
        raise ValueError(f"No rows found for split: {HOLD_SPLIT}")

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Test:", len(test_df))
    print("Hold:", len(hold_df))

    print("\nTrain label distribution:")
    print(train_df["damage_label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["damage_label"].value_counts())

    print("\nHold label distribution:")
    print(hold_df["damage_label"].value_counts())

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": False,
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        XViewBuildingDataset(train_df, train=True),
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        XViewBuildingDataset(test_df, train=False),
        shuffle=False,
        **loader_kwargs,
    )

    hold_loader = DataLoader(
        XViewBuildingDataset(hold_df, train=False),
        shuffle=False,
        **loader_kwargs,
    )

    device = get_device()
    print(f"\nUsing device: {device}")

    print("\nLoading pretrained beta TCVAE encoder...")
    encoder_checkpoint = torch.load(ENCODER_PATH, map_location=device)

    latent_dim = encoder_checkpoint.get("latent_dim", LATENT_DIM)

    encoder = BetaTCVAEEncoder(
        latent_dim=latent_dim,
        use_mu_only=USE_MU_ONLY,
    )

    encoder.encoder.load_state_dict(encoder_checkpoint["encoder"])
    encoder.fc_mu.load_state_dict(encoder_checkpoint["fc_mu"])
    encoder.fc_logvar.load_state_dict(encoder_checkpoint["fc_logvar"])
    encoder = encoder.to(device)

    if FREEZE_ENCODER:
        print("\nFreezing encoder parameters.")
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        print("\nFine tuning encoder parameters.")

    classifier_input_dim = latent_dim if USE_MU_ONLY else latent_dim * 2

    classifier = LatentClassifier(
        input_dim=classifier_input_dim,
        num_classes=4,
    ).to(device)

    model = BetaTCVAEClassifier(encoder, classifier).to(device)

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2, 3]),
            y=train_df["damage_label"].map(LABEL_TO_IDX).values,
        )

        if USE_SQRT_CLASS_WEIGHTS:
            class_weights = np.sqrt(class_weights)

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
        ).to(device)

        print("\nUsing class weights:", class_weights.cpu().numpy())
    else:
        class_weights = None
        print("\nNot using class weights.")

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA)
        print("\nUsing corrected focal loss")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\nUsing cross entropy loss")

    trainable_encoder_params = [
        p for p in model.encoder.parameters() if p.requires_grad
    ]

    optimizer_param_groups = []

    if len(trainable_encoder_params) > 0:
        optimizer_param_groups.append(
            {
                "params": trainable_encoder_params,
                "lr": ENCODER_LR,
            }
        )

    optimizer_param_groups.append(
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR,
        }
    )

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_test_f1 = -1.0
    history = []

    print("\nStarting fixed beta TCVAE classifier training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)

        for x, y in progress:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during classifier training.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader.dataset)

        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            desc=f"Test eval epoch {epoch}",
        )

        epoch_minutes = (time.time() - epoch_start) / 60.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_metrics["loss"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_f1_no_damage": float(test_metrics["per_class_f1"][0]),
                "test_f1_minor": float(test_metrics["per_class_f1"][1]),
                "test_f1_major": float(test_metrics["per_class_f1"][2]),
                "test_f1_destroyed": float(test_metrics["per_class_f1"][3]),
                "epoch_minutes": epoch_minutes,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Macro F1: {test_metrics['macro_f1']:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

        if test_metrics["macro_f1"] > best_test_f1:
            best_test_f1 = test_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    torch.save(best_state, OUTPUT_DIR / "best_model.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    print("\nEvaluating best fixed beta TCVAE classifier...")
    model.load_state_dict(best_state)

    final_test = evaluate(
        model,
        test_loader,
        criterion,
        device,
        desc="Final TEST evaluation",
    )

    final_hold = evaluate(
        model,
        hold_loader,
        criterion,
        device,
        desc="Final HOLD evaluation",
    )

    print(f"\nFinal TEST Macro F1: {final_test['macro_f1']:.4f}")
    print(
        classification_report(
            final_test["targets"],
            final_test["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
            zero_division=0,
        )
    )

    print_prediction_distribution(final_test, "TEST")

    print(f"\nFinal HOLD Macro F1: {final_hold['macro_f1']:.4f}")
    print(
        classification_report(
            final_hold["targets"],
            final_hold["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
            zero_division=0,
        )
    )

    print_prediction_distribution(final_hold, "HOLD")

    test_report = classification_report(
        final_test["targets"],
        final_test["preds"],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    hold_report = classification_report(
        final_hold["targets"],
        final_hold["preds"],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    test_cm = confusion_matrix(
        final_test["targets"],
        final_test["preds"],
        labels=[0, 1, 2, 3],
    )

    hold_cm = confusion_matrix(
        final_hold["targets"],
        final_hold["preds"],
        labels=[0, 1, 2, 3],
    )

    np.save(OUTPUT_DIR / "beta_tcvae_test_preds.npy", np.array(final_test["preds"]))
    np.save(OUTPUT_DIR / "beta_tcvae_test_targets.npy", np.array(final_test["targets"]))
    np.save(OUTPUT_DIR / "beta_tcvae_hold_preds.npy", np.array(final_hold["preds"]))
    np.save(OUTPUT_DIR / "beta_tcvae_hold_targets.npy", np.array(final_hold["targets"]))
    np.save(OUTPUT_DIR / "beta_tcvae_test_confusion_matrix.npy", test_cm)
    np.save(OUTPUT_DIR / "beta_tcvae_hold_confusion_matrix.npy", hold_cm)

    summary = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "classifier_learning_rate": CLASSIFIER_LR,
        "encoder_learning_rate": ENCODER_LR,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "use_sqrt_class_weights": USE_SQRT_CLASS_WEIGHTS,
        "use_focal_loss": USE_FOCAL_LOSS,
        "focal_gamma": FOCAL_GAMMA,
        "freeze_encoder": FREEZE_ENCODER,
        "use_mu_only": USE_MU_ONLY,
        "latent_dim": latent_dim,
        "classifier_input_dim": classifier_input_dim,
        "encoder_path": str(ENCODER_PATH),
        "device": str(device),
        "best_test_macro_f1": float(best_test_f1),
        "final_test_macro_f1": float(final_test["macro_f1"]),
        "final_hold_macro_f1": float(final_hold["macro_f1"]),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "hold_size": int(len(hold_df)),
        "test_prediction_distribution": {
            IDX_TO_LABEL[int(k)]: int(v) for k, v in final_test["pred_counts"].items()
        },
        "hold_prediction_distribution": {
            IDX_TO_LABEL[int(k)]: int(v) for k, v in final_hold["pred_counts"].items()
        },
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    with open(OUTPUT_DIR / "hold_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(hold_report, f, indent=2)

    print("\nSaved fixed beta TCVAE classifier outputs:")
    print(OUTPUT_DIR / "best_model.pt")
    print(OUTPUT_DIR / "training_history.csv")
    print(OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
