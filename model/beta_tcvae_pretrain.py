##############
# beta_tcvae_pretrain_fixed.py
##############

from pathlib import Path
import random
import json
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


##############
# 1. Configuration
##############

BASE_DIR = Path.home() / "Desktop"
CSV_PATH = BASE_DIR / "OOD_processed" / "buildings_all_OOD_with_crops.csv"
OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "beta_tcvae_fixed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

LATENT_DIM = 128
BETA_TC = 1.0

IMAGE_CHANNELS = 6
IMAGE_SIZE = 128

TRAIN_SPLIT = "OOD_train"


##############
# 2. Reproducibility
##############

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


##############
# 3. Device
##############

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


##############
# 4. Dataset
##############

class XViewCropDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        return torch.tensor(x, dtype=torch.float32)


##############
# 5. Model
##############

class BetaTCVAE(nn.Module):
    def __init__(self, latent_dim=128):
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

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 6, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


##############
# 6. Loss helpers
##############

def gaussian_log_density(z, mu, logvar):
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    normalization = -0.5 * (np.log(2.0 * np.pi) + logvar)
    inv_var = torch.exp(-logvar)
    return normalization - 0.5 * ((z - mu) ** 2 * inv_var)


def estimate_total_correlation(z, mu, logvar):
    batch_size, latent_dim = z.shape

    z_expand = z.unsqueeze(1)
    mu_expand = mu.unsqueeze(0)
    logvar_expand = logvar.unsqueeze(0)

    log_q_z_prob = gaussian_log_density(z_expand, mu_expand, logvar_expand)

    log_q_z = torch.logsumexp(log_q_z_prob.sum(dim=2), dim=1) - np.log(batch_size)

    log_q_z_product = torch.logsumexp(log_q_z_prob, dim=1) - np.log(batch_size)
    log_q_z_product = log_q_z_product.sum(dim=1)

    return (log_q_z - log_q_z_product).mean()


def beta_tcvae_loss(recon, x, mu, logvar, z, beta_tc):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    kl_loss = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp()
    ) / x.size(0)

    tc_loss = estimate_total_correlation(z, mu, logvar)

    total_loss = recon_loss + kl_loss + beta_tc * tc_loss

    return total_loss, recon_loss, kl_loss, tc_loss


##############
# 7. Main
##############

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    train_df = df[df["split"] == TRAIN_SPLIT].copy()

    if train_df.empty:
        raise ValueError(f"No training rows found with split == {TRAIN_SPLIT}.")

    print("Training samples:", len(train_df))

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": False,
        "shuffle": True,
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        XViewCropDataset(train_df),
        **loader_kwargs,
    )

    device = get_device()
    print(f"Using device: {device}")

    model = BetaTCVAE(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []

    print("Starting fixed beta TCVAE pretraining...")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_start = time.time()

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        tc_loss_sum = 0.0
        n_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)

        for x in progress:
            x = x.to(device)

            optimizer.zero_grad()

            recon, mu, logvar, z = model(x)

            loss, recon_loss, kl_loss, tc_loss = beta_tcvae_loss(
                recon, x, mu, logvar, z, BETA_TC
            )

            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during beta TCVAE pretraining.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_sum += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            tc_loss_sum += tc_loss.item()
            n_batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.2f}",
                recon=f"{recon_loss.item():.2f}",
                kl=f"{kl_loss.item():.2f}",
                tc=f"{tc_loss.item():.4f}",
            )

        epoch_minutes = (time.time() - epoch_start) / 60.0

        record = {
            "epoch": epoch,
            "loss": total_loss_sum / n_batches,
            "reconstruction_loss": recon_loss_sum / n_batches,
            "kl_loss": kl_loss_sum / n_batches,
            "tc_loss": tc_loss_sum / n_batches,
            "epoch_minutes": epoch_minutes,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {record['loss']:.2f} | "
            f"Recon: {record['reconstruction_loss']:.2f} | "
            f"KL: {record['kl_loss']:.2f} | "
            f"TC: {record['tc_loss']:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

        torch.save(model.state_dict(), OUTPUT_DIR / "beta_tcvae_latest.pt")

    torch.save(model.state_dict(), OUTPUT_DIR / "beta_tcvae_final.pt")

    encoder_state = {
        "encoder": model.encoder.state_dict(),
        "fc_mu": model.fc_mu.state_dict(),
        "fc_logvar": model.fc_logvar.state_dict(),
        "latent_dim": LATENT_DIM,
        "use_mu_only_for_classifier": True,
    }

    torch.save(encoder_state, OUTPUT_DIR / "beta_tcvae_encoder.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "beta_tcvae_training_history.csv", index=False)

    summary = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "latent_dim": LATENT_DIM,
        "beta_tc": BETA_TC,
        "train_size": int(len(train_df)),
        "device": str(device),
        "notes": "Fixed version. Pretrains encoder, saves mu representation for downstream classifier. Logvar is not used as classifier input.",
    }

    with open(OUTPUT_DIR / "beta_tcvae_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved fixed beta TCVAE model outputs:")
    print(OUTPUT_DIR / "beta_tcvae_final.pt")
    print(OUTPUT_DIR / "beta_tcvae_encoder.pt")
    print(OUTPUT_DIR / "beta_tcvae_training_history.csv")


if __name__ == "__main__":
    main()
