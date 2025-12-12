import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "model_ready"

ALL_SITES_CSV = DATA_DIR / "all_sites.csv"

OUT_TRAIN = DATA_DIR / "windows_train.npz"
OUT_VAL   = DATA_DIR / "windows_val.npz"
OUT_TEST  = DATA_DIR / "windows_test.npz"

# window lengths
CONTEXT_LENGTH = 96   # past context length
PRED_LENGTH    = 24   # prediction horizon

# feature columns (normalized)
FEAT_COLS = [
    "ac_power_kw_norm",
    "poa_irradiance_norm",
    "module_temp_norm",
]

TARGET_COL = "ac_power_kw_norm"  # what we’re forecasting (normalized)


# ---------------- WINDOW LOGIC ----------------

def make_windows_for_site(site_df: pd.DataFrame, site_id: int):
    """
    Given a single site's time series (already in time order),
    build sliding windows and split into train/val/test for that site.

    Returns:
        (X_train, y_train, ids_train,
         X_val,   y_val,   ids_val,
         X_test,  y_test,  ids_test)
    """
    # ensure sorted as they appear; we're assuming all_sites was already time-sorted
    site_df = site_df.reset_index(drop=True)

    # extract features and target as float32 arrays
    feats = site_df[FEAT_COLS].to_numpy(dtype="float32")       # shape (n, F)
    target = site_df[TARGET_COL].to_numpy(dtype="float32")     # shape (n,)

    n = len(site_df)
    total_len = CONTEXT_LENGTH + PRED_LENGTH

    if n < total_len:
        # not enough data to form a single window
        return ([], [], [], [], [], [], [], [], [])

    # index-based split boundaries
    train_last_idx = int(n * 0.7) - 1
    val_last_idx   = int(n * 0.85) - 1

    X_train, y_train, ids_train = [], [], []
    X_val,   y_val,   ids_val   = [], [], []
    X_test,  y_test,  ids_test  = [], [], []

    # max start index so that end_idx stays in-bounds
    max_start = n - total_len + 1

    for t in range(max_start):
        end_idx = t + total_len - 1  # index of last target step

        # decide which split this window belongs to
        if end_idx <= train_last_idx:
            bucket = "train"
        elif end_idx <= val_last_idx:
            bucket = "val"
        else:
            bucket = "test"

        # build windows
        X_t = feats[t : t + CONTEXT_LENGTH, :]  # (C, F)
        y_t = target[t + CONTEXT_LENGTH : t + total_len]  # (P,)

        if bucket == "train":
            X_train.append(X_t)
            y_train.append(y_t)
            ids_train.append(site_id)
        elif bucket == "val":
            X_val.append(X_t)
            y_val.append(y_t)
            ids_val.append(site_id)
        else:
            X_test.append(X_t)
            y_test.append(y_t)
            ids_test.append(site_id)

    return (
        X_train, y_train, ids_train,
        X_val,   y_val,   ids_val,
        X_test,  y_test,  ids_test,
    )


def main():
    if not ALL_SITES_CSV.exists():
        raise FileNotFoundError(f"all_sites.csv not found at {ALL_SITES_CSV}")

    df = pd.read_csv(ALL_SITES_CSV)

    # sanity check columns
    missing = [c for c in FEAT_COLS + [TARGET_COL, "site_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in all_sites.csv: {missing}")

    # containers for concatenated splits across all sites
    X_tr_all, y_tr_all, ids_tr_all = [], [], []
    X_va_all, y_va_all, ids_va_all = [], [], []
    X_te_all, y_te_all, ids_te_all = [], [], []

    # group by site_id in the order they appear
    for site_id, site_df in df.groupby("site_id", sort=False):
        print(f"[SITE {site_id}] rows: {len(site_df)}")

        (
            X_tr, y_tr, ids_tr,
            X_va, y_va, ids_va,
            X_te, y_te, ids_te,
        ) = make_windows_for_site(site_df, site_id)

        print(
            f"  windows -> train: {len(X_tr)}, "
            f"val: {len(X_va)}, test: {len(X_te)}"
        )

        X_tr_all.extend(X_tr)
        y_tr_all.extend(y_tr)
        ids_tr_all.extend(ids_tr)

        X_va_all.extend(X_va)
        y_va_all.extend(y_va)
        ids_va_all.extend(ids_va)

        X_te_all.extend(X_te)
        y_te_all.extend(y_te)
        ids_te_all.extend(ids_te)

    # convert lists to arrays
    def to_array(list_X, list_y, list_ids):
        if not list_X:
            return (
                np.empty((0, CONTEXT_LENGTH, len(FEAT_COLS)), dtype="float32"),
                np.empty((0, PRED_LENGTH), dtype="float32"),
                np.empty((0,), dtype="int64"),
            )
        X = np.stack(list_X, axis=0)
        y = np.stack(list_y, axis=0)
        ids = np.asarray(list_ids, dtype="int64")
        return X, y, ids

    X_tr, y_tr, ids_tr = to_array(X_tr_all, y_tr_all, ids_tr_all)
    X_va, y_va, ids_va = to_array(X_va_all, y_va_all, ids_va_all)
    X_te, y_te, ids_te = to_array(X_te_all, y_te_all, ids_te_all)

    print("\nFinal window counts:")
    print(f"  train: X {X_tr.shape}, y {y_tr.shape}")
    print(f"  val:   X {X_va.shape}, y {y_va.shape}")
    print(f"  test:  X {X_te.shape}, y {y_te.shape}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUT_TRAIN,
        X=X_tr,
        y=y_tr,
        site_id=ids_tr,
    )
    np.savez_compressed(
        OUT_VAL,
        X=X_va,
        y=y_va,
        site_id=ids_va,
    )
    np.savez_compressed(
        OUT_TEST,
        X=X_te,
        y=y_te,
        site_id=ids_te,
    )

    print(f"\nSaved:")
    print(f"  {OUT_TRAIN}")
    print(f"  {OUT_VAL}")
    print(f"  {OUT_TEST}")


if __name__ == "__main__":
    main()
