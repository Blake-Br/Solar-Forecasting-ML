import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
)

# ---------------- PATHS / CONSTANTS ----------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "model_ready"

TRAIN_PATH = DATA_DIR / "windows_train.npz"
VAL_PATH   = DATA_DIR / "windows_val.npz"

MODEL_DIR  = BASE_DIR / "models" / "ts_transformer"

# Must match make_windows.py
CONTEXT_LENGTH = 96
PRED_LENGTH    = 24

INPUT_SIZE = 1   # we use only ac_power_kw_norm (channel 0)


# ---------------- DATASET ----------------

class WindowDataset(Dataset):
    """
    Wraps the .npz window files into a PyTorch Dataset for HF TimeSeriesTransformer.
    We use:
      - past_values: history of ac_power_kw_norm (CONTEXT_LENGTH, 1)
      - future_values: future ac_power_kw_norm (PRED_LENGTH, 1)
      - static_categorical_features: site_id (1,)
    """

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)

        # (N, C, F)
        self.X = data["X"].astype("float32")
        # (N, PRED_LENGTH)
        self.y = data["y"].astype("float32")
        # (N,)
        self.site_id = data["site_id"].astype("int64")

        # sanity checks
        assert self.X.ndim == 3, self.X.shape
        assert self.y.ndim == 2, self.y.shape
        assert self.X.shape[0] == self.y.shape[0] == self.site_id.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Take only channel 0 = ac_power_kw_norm
        past_vals = self.X[idx, :, 0]          # (C,)
        future_vals = self.y[idx, :]          # (P,)

        # reshape to (C, 1) and (P, 1) for input_size = 1
        past_vals = past_vals[:, None]
        future_vals = future_vals[:, None]

        site = self.site_id[idx]

        return {
            "past_values": torch.from_numpy(past_vals),       # (C, 1)
            "future_values": torch.from_numpy(future_vals),   # (P, 1)
            "static_categorical_features": torch.tensor([site], dtype=torch.long),  # (1,)
        }


# ---------------- HELPER: LOAD NPZ TO GET CARDINALITY ----------------

def get_num_sites(npz_path: Path) -> int:
    data = np.load(npz_path)
    site_ids = data["site_id"]
    return int(np.unique(site_ids).shape[0])


# ---------------- MAIN TRAINING ----------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # how many unique sites
    num_sites = get_num_sites(TRAIN_PATH)
    print("Number of sites:", num_sites)

    # Datasets + loaders
    train_ds = WindowDataset(TRAIN_PATH)
    val_ds   = WindowDataset(VAL_PATH)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    # HF TimeSeriesTransformer config
    config = TimeSeriesTransformerConfig(
        prediction_length=PRED_LENGTH,
        context_length=CONTEXT_LENGTH,
        input_size=INPUT_SIZE,             # 1D series (ac_power_kw_norm)
        num_time_features=0,               # we’re not feeding extra time features here
        num_static_categorical_features=1, # site_id
        num_static_real_features=0,
        # site embedding cardinality
        cardinality=[num_sites],
        # basic model size; keep small for speed
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.1,
    )

    model = TimeSeriesTransformerForPrediction(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    EPOCHS = 5

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # -------- TRAIN --------
        model.train()
        total_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            past_values = batch["past_values"].to(device)                   # (B, C, 1)
            future_values = batch["future_values"].to(device)               # (B, P, 1)
            static_cat = batch["static_categorical_features"].to(device)    # (B, 1)

            optimizer.zero_grad()

            outputs = model(
                past_values=past_values,
                future_values=future_values,
                static_categorical_features=static_cat,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        avg_train_loss = total_loss / max(n_steps, 1)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                past_values = batch["past_values"].to(device)
                future_values = batch["future_values"].to(device)
                static_cat = batch["static_categorical_features"].to(device)

                outputs = model(
                    past_values=past_values,
                    future_values=future_values,
                    static_categorical_features=static_cat,
                )
                loss = outputs.loss

                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(val_steps, 1)

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"- train loss: {avg_train_loss:.4f} "
            f"- val loss: {avg_val_loss:.4f}"
        )

        # Optional: save checkpoint per epoch
        epoch_dir = MODEL_DIR / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        config.save_pretrained(epoch_dir)

    # Final save
    model.save_pretrained(MODEL_DIR)
    config.save_pretrained(MODEL_DIR)
    print("Saved final model to", MODEL_DIR)


if __name__ == "__main__":
    train()
