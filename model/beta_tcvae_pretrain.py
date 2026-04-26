##############
# 1. Imports
##############
# We import the libraries needed for:
# - file paths
# - reproducibility
# - data handling
# - model training
# - progress bars
# - PyTorch neural network components

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
# 2. Configuration
##############
# This section defines all important settings in one place.
# This makes the experiment easier to reproduce and modify.

BASE_DIR = Path.home() / "Desktop"
CSV_PATH = BASE_DIR / "OOD_processed" / "buildings_all_OOD_with_crops.csv"
OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "beta_tcvae"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

LATENT_DIM = 64
BETA_TC = 6.0

IMAGE_CHANNELS = 6
IMAGE_SIZE = 128


##############
# 3. Reproducibility
##############
# This fixes random seeds so that the experiment is as reproducible as possible.
# It affects:
# - Python random numbers
# - NumPy random numbers
# - PyTorch random numbers

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


##############
# 4. Device selection
##############
# This selects the fastest available device.
# On your MacBook, this should choose "mps", which uses Apple Silicon acceleration.

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


##############
# 5. Dataset class
##############
# This dataset reads the saved crop files from disk.
# Each crop is stored as a .npy file with shape:
# (128, 128, 6)
#
# The 6 channels are:
# - first 3 channels: pre-disaster RGB image
# - last 3 channels: post-disaster RGB image
#
# The data is converted to:
# (6, 128, 128)
# because PyTorch expects channel-first format.

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
# 6. β-TCVAE model
##############
# The model has two main parts:
#
# 1. Encoder:
#    compresses the input image into a smaller latent representation.
#
# 2. Decoder:
#    reconstructs the original image from the latent representation.
#
# The latent representation is described by:
# - mu: mean of the latent distribution
# - logvar: log variance of the latent distribution
#
# A latent vector z is sampled using the reparameterization trick.

class BetaTCVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        ##############
        # Encoder
        ##############
        # Input:
        # 6 x 128 x 128
        #
        # Output after convolutions:
        # 256 x 8 x 8
        #
        # This is then flattened and mapped to:
        # - mu
        # - logvar

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

        ##############
        # Decoder
        ##############
        # The decoder takes the latent vector z and reconstructs the image.
        #
        # Input:
        # latent vector z
        #
        # Output:
        # reconstructed 6 x 128 x 128 image

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

    ##############
    # Encode
    ##############
    # Converts an image into mu and logvar.

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    ##############
    # Reparameterization trick
    ##############
    # Instead of sampling z directly in a way that blocks gradients,
    # we sample random noise and scale it using mu and logvar.
    #
    # This allows the model to learn through backpropagation.

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    ##############
    # Decode
    ##############
    # Converts latent vector z back into a reconstructed image.

    def decode(self, z):
        h = self.decoder_input(z)
        recon = self.decoder(h)
        return recon

    ##############
    # Forward pass
    ##############
    # Full β-TCVAE process:
    # image -> encoder -> mu/logvar -> sample z -> decoder -> reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


##############
# 7. Gaussian log density
##############
# This helper function computes how likely a latent vector is under
# a Gaussian distribution.
#
# It is used to estimate Total Correlation.

def gaussian_log_density(z, mu, logvar):
    normalization = -0.5 * (np.log(2 * np.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((z - mu) ** 2 * inv_var)
    return log_density


##############
# 8. Total Correlation estimate
##############
# Total Correlation checks whether latent variables are independent.
#
# If dimensions of z are highly dependent on each other,
# the TC value becomes larger.
#
# β-TCVAE penalizes this, encouraging the model to learn more separated factors.

def estimate_total_correlation(z, mu, logvar):
    batch_size, latent_dim = z.shape

    z_expand = z.unsqueeze(1)
    mu_expand = mu.unsqueeze(0)
    logvar_expand = logvar.unsqueeze(0)

    log_q_z_prob = gaussian_log_density(z_expand, mu_expand, logvar_expand)

    log_q_z = torch.logsumexp(log_q_z_prob.sum(dim=2), dim=1) - np.log(batch_size)

    log_q_z_product = torch.logsumexp(log_q_z_prob, dim=1) - np.log(batch_size)
    log_q_z_product = log_q_z_product.sum(dim=1)

    tc = (log_q_z - log_q_z_product).mean()
    return tc


##############
# 9. β-TCVAE loss
##############
# The total loss has three parts:
#
# 1. Reconstruction loss:
#    makes the reconstructed image similar to the input image.
#
# 2. KL loss:
#    regularizes the latent space.
#
# 3. Total Correlation loss:
#    encourages latent variables to be independent.
#
# The β value controls how strongly TC is penalized.

def beta_tcvae_loss(recon, x, mu, logvar, z, beta_tc):
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / x.size(0)

    tc_loss = estimate_total_correlation(z, mu, logvar)

    total_loss = recon_loss + kl_loss + beta_tc * tc_loss

    return total_loss, recon_loss, kl_loss, tc_loss


##############
# 10. Main training function
##############
# This function:
# - loads the crop metadata
# - keeps only the train split
# - creates the DataLoader
# - trains the β-TCVAE
# - saves the trained model and encoder

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    train_df = df[df["split"] == "OOD_train"].copy()

    if train_df.empty:
        raise ValueError("No training rows found with split == 'OOD_train'.")

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

    print("Starting β-TCVAE pretraining...")

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

            loss.backward()
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


##############
# 11. Save final outputs
##############
# After training, the script saves:
#
# - the full β-TCVAE model
# - the encoder only
# - training history
# - experiment summary
#
# The encoder is saved separately because it will be reused later
# as a feature extractor for damage classification.

    torch.save(model.state_dict(), OUTPUT_DIR / "beta_tcvae_final.pt")

    encoder_state = {
        "encoder": model.encoder.state_dict(),
        "fc_mu": model.fc_mu.state_dict(),
        "fc_logvar": model.fc_logvar.state_dict(),
        "latent_dim": LATENT_DIM,
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
    }

    with open(OUTPUT_DIR / "beta_tcvae_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved β-TCVAE model outputs:")
    print(OUTPUT_DIR / "beta_tcvae_final.pt")
    print(OUTPUT_DIR / "beta_tcvae_encoder.pt")
    print(OUTPUT_DIR / "beta_tcvae_training_history.csv")


##############
# 12. Run script
##############
# This ensures the script only runs when executed directly,
# not when imported into another file.

if __name__ == "__main__":
    main()