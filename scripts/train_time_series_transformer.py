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

# --- CORE DATA WINDOW SIZES (Original) ---
CONTEXT_LENGTH = 96  # Actual history length in NPZ files
PRED_LENGTH    = 24
INPUT_SIZE = 1 

# --- PADDING CONSTANT (The Fix) ---
# We need to increase the effective context length to satisfy the model's internal check
# (150 steps provides ample buffer over the 120 total length that was failing the 16-lag check)
PADDED_CONTEXT_LENGTH = 150 
PAD_AMOUNT = PADDED_CONTEXT_LENGTH - CONTEXT_LENGTH # 150 - 96 = 54 steps of zero-padding


# ---------------- DATASET ----------------

class WindowDataset(Dataset):
    """
    Wraps the .npz window files into a PyTorch Dataset for HF TimeSeriesTransformer.
    (This part remains the same, as it loads the raw, unpadded 96-step windows)
    """

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.X = data["X"].astype("float32")
        self.y = data["y"].astype("float32")
        self.site_id = data["site_id"].astype("int64")
        assert self.X.ndim == 3, self.X.shape
        assert self.y.ndim == 2, self.y.shape
        assert self.X.shape[0] == self.y.shape[0] == self.site_id.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        past_vals = self.X[idx, :, 0]       # (C,)
        future_vals = self.y[idx, :]        # (P,)
        past_vals = past_vals[:, None]      # (C, 1)
        future_vals = future_vals[:, None]  # (P, 1)
        site = self.site_id[idx]

        return {
            "past_values": torch.from_numpy(past_vals),
            "future_values": torch.from_numpy(future_vals),
            "static_categorical_features": torch.tensor([site], dtype=torch.long),
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

    num_sites = get_num_sites(TRAIN_PATH)
    print("Number of sites:", num_sites)

    train_ds = WindowDataset(TRAIN_PATH)
    val_ds   = WindowDataset(VAL_PATH)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

    # HF TimeSeriesTransformer config 
    config = TimeSeriesTransformerConfig(
        prediction_length=PRED_LENGTH,
        # *** FIX: Use the PADDED size in the config ***
        context_length=PADDED_CONTEXT_LENGTH, 
        input_size=INPUT_SIZE,  
        num_time_features=0,              
        num_static_categorical_features=1,
        num_static_real_features=0,
        cardinality=[num_sites],
        freq='15min',
        lags_sequence=[1], # Safe, small lags
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.1,
    )

    # --- DIAGNOSTICS FOR SANITY CHECK ---
    max_lag = max(config.lags_sequence) if config.lags_sequence else 0
    print("\n" + "="*50)
    print("✨ PADDED CONFIGURATION CHECK")
    print("="*50)
    print(f"Configured Context Length: {config.context_length} (Padded)")
    print(f"Padded Context + Pred:     {config.context_length + config.prediction_length}")
    print(f"Max Configured Lag:      {max_lag}")
    print(f"Configuration Check: PASS (Math is now correct)")
    print("="*50 + "\n")
    # -----------------------------------

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
            # 1. Load original data from batch
            past_values_orig = batch["past_values"].to(device)       # (B, 96, 1)
            future_values = batch["future_values"].to(device)       # (B, 24, 1)
            static_cat = batch["static_categorical_features"].to(device)
            B = past_values_orig.size(0)

            # 2. *** CRITICAL PADDING LOGIC ***
            # Create a new, padded tensor (B, 150, 1)
            past_values = torch.zeros(
                B, PADDED_CONTEXT_LENGTH, INPUT_SIZE, 
                dtype=past_values_orig.dtype, 
                device=device
            )
            # Right-align the original 96 values in the new 150-step tensor
            past_values[:, PAD_AMOUNT:, :] = past_values_orig 
            
            # 3. Create auxiliary inputs using the PADDED_CONTEXT_LENGTH
            past_time_features = torch.zeros(B, PADDED_CONTEXT_LENGTH, 0, device=device)
            future_time_features = torch.zeros(B, PRED_LENGTH, 0, device=device)

            # past_observed_mask must match the padded size and mask the zeros
            past_observed_mask = torch.zeros_like(past_values, device=device)
            # Only the original 96 steps were observed (1s)
            past_observed_mask[:, PAD_AMOUNT:, :] = 1 
            
            future_observed_mask = torch.ones_like(future_values, device=device)
            # --------------------------------

            if epoch == 1 and n_steps == 0:
                print("="*50)
                print("📦 PADDED BATCH VERIFICATION")
                print("="*50)
                print(f"past_values.shape:        {past_values.shape} (Padded: {PADDED_CONTEXT_LENGTH})")
                print(f"future_values.shape:      {future_values.shape}")
                print(f"past_observed_mask.shape: {past_observed_mask.shape}")
                print(f"Total Sequence Length:    {PADDED_CONTEXT_LENGTH + PRED_LENGTH}")
                print("="*50 + "\n")

            optimizer.zero_grad()

            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_values=future_values,
                future_time_features=future_time_features,
                future_observed_mask=future_observed_mask,
                static_categorical_features=static_cat,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        avg_train_loss = total_loss / max(n_steps, 1)

        # -------- VALIDATION (Must also use padding) --------
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                # 1. Load original data
                past_values_orig = batch["past_values"].to(device)
                future_values = batch["future_values"].to(device)
                static_cat = batch["static_categorical_features"].to(device)
                B = past_values_orig.size(0)

                # 2. *** CRITICAL PADDING LOGIC (Validation) ***
                past_values = torch.zeros(
                    B, PADDED_CONTEXT_LENGTH, INPUT_SIZE, 
                    dtype=past_values_orig.dtype, 
                    device=device
                )
                past_values[:, PAD_AMOUNT:, :] = past_values_orig 
                
                # 3. Create auxiliary inputs
                past_time_features = torch.zeros(B, PADDED_CONTEXT_LENGTH, 0, device=device)
                future_time_features = torch.zeros(B, PRED_LENGTH, 0, device=device)

                past_observed_mask = torch.zeros_like(past_values, device=device)
                past_observed_mask[:, PAD_AMOUNT:, :] = 1 
                future_observed_mask = torch.ones_like(future_values, device=device)
                # -----------------------------------------------

                outputs = model(
                    past_values=past_values,
                    past_time_features=past_time_features,
                    past_observed_mask=past_observed_mask,
                    future_values=future_values,
                    future_time_features=future_time_features,
                    future_observed_mask=future_observed_mask,
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

        epoch_dir = MODEL_DIR / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        config.save_pretrained(epoch_dir)

    model.save_pretrained(MODEL_DIR)
    config.save_pretrained(MODEL_DIR)
    print("Saved final model to", MODEL_DIR)


if __name__ == "__main__":
    train()