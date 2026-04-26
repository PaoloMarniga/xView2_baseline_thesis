##############
# 1. Imports
##############
# This script trains a supervised damage classifier using the representation
# learned by the β-TCVAE.
#
# The β-TCVAE was already trained in a previous step. That model learned how
# to compress each 6-channel building crop into a smaller latent vector.
#
# In this script, we do NOT train the β-TCVAE again.
# Instead, we:
#
# 1. Load the trained β-TCVAE encoder
# 2. Freeze its weights
# 3. Use it to convert each crop into a latent representation
# 4. Train a small classifier on top of that latent representation
#
# This allows us to test whether the representation learned by β-TCVAE is
# useful for damage classification.

from pathlib import Path
import random
import json
import copy
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


##############
# 2. Configuration
##############
# This section contains all important experiment settings.
#
# Keeping them in one place makes the script easier to reproduce and modify.
#
# CSV_PATH points to the OOD dataset with crop paths.
#
# ENCODER_PATH points to the encoder saved after β-TCVAE pretraining.
#
# OUTPUT_DIR is where this classifier script will save:
# - the trained classifier
# - training history
# - predictions
# - classification reports
# - summary files

BASE_DIR = Path.home() / "Desktop"

CSV_PATH = BASE_DIR / "OOD_processed" / "buildings_all_OOD_with_crops.csv"

ENCODER_PATH = BASE_DIR / "OOD_training_outputs" / "beta_tcvae" / "beta_tcvae_encoder.pt"

OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "beta_tcvae_classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


##############
# 3. Hyperparameters
##############
# These control training behavior.
#
# BATCH_SIZE:
# Number of crops processed at once.
#
# NUM_EPOCHS:
# Number of full passes over the training set.
#
# LEARNING_RATE:
# Controls how much the classifier weights are updated at each step.
#
# NUM_WORKERS:
# Controls parallel data loading.
#
# USE_CLASS_WEIGHTS:
# If True, the loss gives more importance to minority classes.
# This is useful because no-damage is much more frequent than the damage classes.

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
USE_CLASS_WEIGHTS = True

LATENT_DIM = 64

TRAIN_SPLIT = "OOD_train"
TEST_SPLIT = "OOD_test"
HOLD_SPLIT = "OOD_hold"


##############
# 4. Label mapping
##############
# The damage labels are strings in the dataframe.
#
# Neural networks need numeric labels, so we map:
#
# no-damage     -> 0
# minor-damage  -> 1
# major-damage  -> 2
# destroyed     -> 3
#
# This mapping must remain identical across all experiments so that results
# are comparable.

LABEL_TO_IDX = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


##############
# 5. Reproducibility
##############
# This fixes random seeds for:
# - Python random module
# - NumPy
# - PyTorch
#
# This does not guarantee perfectly identical results on every machine,
# especially with GPU or MPS acceleration, but it makes the experiment
# much more reproducible.

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


##############
# 6. Device selection
##############
# This function selects the fastest available hardware.
#
# Priority:
# 1. CUDA, if using an NVIDIA GPU
# 2. MPS, if using Apple Silicon acceleration
# 3. CPU, if no accelerator is available
#
# On your MacBook Air M4, this should select "mps".

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


##############
# 7. Dataset class
##############
# This dataset reads one crop at a time from the crop_path column.
#
# Each crop is stored as a .npy file with shape:
#
# (128, 128, 6)
#
# The 6 channels are:
#
# channels 0-2: pre-disaster RGB crop
# channels 3-5: post-disaster RGB crop
#
# The crop is converted to:
#
# (6, 128, 128)
#
# because PyTorch expects channel-first input.
#
# The dataset returns:
#
# x = image tensor
# y = numeric damage label

class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        y = LABEL_TO_IDX[row["damage_label"]]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


##############
# 8. β-TCVAE encoder architecture
##############
# This class rebuilds the encoder architecture used during β-TCVAE pretraining.
#
# It must match the original encoder exactly, otherwise the saved weights
# cannot be loaded correctly.
#
# The encoder takes:
#
# 6 x 128 x 128 crop
#
# and compresses it through convolutional layers into a latent vector.
#
# The original β-TCVAE encoder produced:
# - mu
# - logvar
#
# For classification, we use mu only.
#
# Why use mu?
# Because mu is the stable deterministic representation of the input.
# Sampling z would add noise, which is useful during VAE training but less
# useful for deterministic classification.

class BetaTCVAEEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

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
        return mu


##############
# 9. Latent classifier
##############
# This is the supervised classifier trained on top of the β-TCVAE latent
# representation.
#
# It receives a latent vector of size 64 and outputs 4 logits, one for each
# damage class.
#
# Structure:
#
# latent vector -> hidden layer -> hidden layer -> 4-class output
#
# Dropout is used to reduce overfitting.

class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=64, num_classes=4):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes),
        )

    def forward(self, z):
        return self.classifier(z)


##############
# 10. Full β-TCVAE classifier model
##############
# This combines:
#
# frozen β-TCVAE encoder + trainable classifier
#
# The encoder is frozen because we want to evaluate the representation learned
# during unsupervised β-TCVAE pretraining.
#
# Only the classifier is trained.
#
# This makes the comparison cleaner:
#
# ResNet50 baseline:
# image -> supervised ResNet features -> damage class
#
# β-TCVAE classifier:
# image -> unsupervised latent features -> damage class

class BetaTCVAEClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)

        logits = self.classifier(z)
        return logits


##############
# 11. Evaluation function
##############
# This function evaluates the model on a dataloader.
#
# It computes:
#
# - average loss
# - macro F1
# - per-class F1
# - predictions
# - true labels
#
# Macro F1 is important because the dataset is imbalanced.
# It gives equal importance to all classes instead of being dominated by
# no-damage.

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
    macro_f1 = f1_score(targets_all, preds_all, average="macro")
    per_class_f1 = f1_score(targets_all, preds_all, average=None, labels=[0, 1, 2, 3])

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "preds": preds_all,
        "targets": targets_all,
    }


##############
# 12. Main function
##############
# This is the full training pipeline.
#
# It performs the following steps:
#
# 1. Load the OOD crop dataframe
# 2. Split it into OOD_train, OOD_test, and OOD_hold
# 3. Build dataloaders
# 4. Load the pretrained β-TCVAE encoder
# 5. Freeze the encoder
# 6. Train only the classifier head
# 7. Select the best model based on OOD_test macro F1
# 8. Evaluate final performance on OOD_test and OOD_hold
# 9. Save outputs for later analysis

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

    ##############
    # 12.1 DataLoaders
    ##############
    # DataLoaders feed batches of crops into the model.
    #
    # The training dataloader uses shuffle=True so the classifier sees
    # samples in a different order each epoch.
    #
    # Test and hold dataloaders use shuffle=False to preserve deterministic
    # evaluation and match prediction order to dataframe order.

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": False,
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        XViewBuildingDataset(train_df),
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        XViewBuildingDataset(test_df),
        shuffle=False,
        **loader_kwargs,
    )

    hold_loader = DataLoader(
        XViewBuildingDataset(hold_df),
        shuffle=False,
        **loader_kwargs,
    )

    ##############
    # 12.2 Device
    ##############

    device = get_device()
    print(f"\nUsing device: {device}")

    ##############
    # 12.3 Load pretrained encoder
    ##############
    # This loads the encoder saved after β-TCVAE pretraining.
    #
    # The checkpoint contains:
    # - encoder convolutional layers
    # - fc_mu layer
    # - fc_logvar layer
    # - latent dimension
    #
    # The classifier uses fc_mu as the latent representation.

    print("\nLoading pretrained β-TCVAE encoder...")
    encoder_checkpoint = torch.load(ENCODER_PATH, map_location=device)

    latent_dim = encoder_checkpoint.get("latent_dim", LATENT_DIM)

    encoder = BetaTCVAEEncoder(latent_dim=latent_dim)

    encoder.encoder.load_state_dict(encoder_checkpoint["encoder"])
    encoder.fc_mu.load_state_dict(encoder_checkpoint["fc_mu"])
    encoder.fc_logvar.load_state_dict(encoder_checkpoint["fc_logvar"])

    encoder = encoder.to(device)

    ##############
    # 12.4 Build full classifier model
    ##############

    classifier = LatentClassifier(latent_dim=latent_dim, num_classes=4).to(device)

    model = BetaTCVAEClassifier(encoder, classifier).to(device)

    ##############
    # 12.5 Loss function
    ##############
    # Cross-entropy is used because this is a 4-class classification task.
    #
    # Class weights are optionally applied to address class imbalance.
    # Since no-damage is the dominant class, minority damage classes receive
    # larger weights.

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2, 3]),
            y=train_df["damage_label"].map(LABEL_TO_IDX).values,
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\nUsing class weights:", class_weights.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()

    ##############
    # 12.6 Optimizer
    ##############
    # Only the classifier parameters are optimized.
    # The encoder remains frozen.

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_test_f1 = -1.0
    history = []

    ##############
    # 12.7 Training loop
    ##############
    # For each epoch:
    #
    # 1. Train classifier on OOD_train
    # 2. Evaluate on OOD_test
    # 3. Save the model if OOD_test macro F1 improves
    #
    # OOD_hold is not used during training or model selection.

    print("\nStarting β-TCVAE classifier training...")

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

            loss.backward()
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
            f"Test F1: {test_metrics['macro_f1']:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

        if test_metrics["macro_f1"] > best_test_f1:
            best_test_f1 = test_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    ##############
    # 12.8 Save best model
    ##############

    torch.save(best_state, OUTPUT_DIR / "best_model.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    ##############
    # 12.9 Final evaluation
    ##############
    # The best model selected on OOD_test is evaluated again on:
    #
    # - OOD_test
    # - OOD_hold
    #
    # The OOD_hold result is the final unbiased evaluation.

    print("\nEvaluating best β-TCVAE classifier...")
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
        )
    )

    print(f"\nFinal HOLD Macro F1: {final_hold['macro_f1']:.4f}")
    print(
        classification_report(
            final_hold["targets"],
            final_hold["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
        )
    )

    ##############
    # 12.10 Save outputs
    ##############
    # Predictions and targets are saved so that confusion matrices and
    # qualitative analysis can be generated later without rerunning the model.

    test_report = classification_report(
        final_test["targets"],
        final_test["preds"],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
    )

    hold_report = classification_report(
        final_hold["targets"],
        final_hold["preds"],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
    )

    np.save(OUTPUT_DIR / "beta_tcvae_test_preds.npy", np.array(final_test["preds"]))
    np.save(OUTPUT_DIR / "beta_tcvae_test_targets.npy", np.array(final_test["targets"]))
    np.save(OUTPUT_DIR / "beta_tcvae_hold_preds.npy", np.array(final_hold["preds"]))
    np.save(OUTPUT_DIR / "beta_tcvae_hold_targets.npy", np.array(final_hold["targets"]))

    summary = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "latent_dim": latent_dim,
        "encoder_path": str(ENCODER_PATH),
        "device": str(device),
        "best_test_macro_f1": float(best_test_f1),
        "final_test_macro_f1": float(final_test["macro_f1"]),
        "final_hold_macro_f1": float(final_hold["macro_f1"]),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "hold_size": int(len(hold_df)),
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    with open(OUTPUT_DIR / "hold_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(hold_report, f, indent=2)

    print("\nSaved β-TCVAE classifier outputs:")
    print(OUTPUT_DIR / "best_model.pt")
    print(OUTPUT_DIR / "training_history.csv")
    print(OUTPUT_DIR / "summary.json")


##############
# 13. Run script
##############
# This ensures that the script only runs when executed directly.

if __name__ == "__main__":
    main()