import re
from pathlib import Path

import pandas as pd

# ------------- CONFIG -------------

BASE_DIR = Path(__file__).resolve().parents[1]

IN_DIR = BASE_DIR / "data" / "processed_final"
OUT_DIR = BASE_DIR / "data" / "model_ready"

# Feature columns in physical units
FEATURE_COLS = ["ac_power_kw", "poa_irradiance", "module_temp"]

# Output filenames
UNIFIED_PATH = OUT_DIR / "all_sites.csv"
STATS_PATH = OUT_DIR / "site_normalization_stats.csv"


# ------------- HELPERS -------------

def load_site_file(path: Path) -> pd.DataFrame:
    """
    Load a single processed_final CSV as a DataFrame with DatetimeIndex.
    Assumes the first column is the datetime index if there is no explicit
    'measured_on'/'timestamp' column.
    """
    # Try to read with first column as index (most likely how you saved it)
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # If index is not datetime, try typical time columns
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["measured_on", "timestamp"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break

    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"[DEBUG] Columns in {path.name}: {df.columns.tolist()}")
        print(f"[DEBUG] First 5 rows:")
        print(df.head())
        print(f"[DEBUG] Index type: {type(df.index)}")

        raise ValueError(f"Could not establish DatetimeIndex for {path}")

    df = df.sort_index()

    # Sanity check for required feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing feature columns: {missing}")

    return df


def infer_site_id(path: Path, df: pd.DataFrame) -> int:
    """
    Determine site_id for a given file.
    Priority:
      1) 'system_id' column if present (single unique value).
      2) Parse from filename pattern like 'system_1234_...csv'.
    """
    if "system_id" in df.columns:
        uniq = df["system_id"].dropna().unique()
        if len(uniq) == 1:
            return int(uniq[0])
        else:
            raise ValueError(f"{path.name}: system_id not unique: {uniq}")

    # Try to parse from filename, e.g. 'system_1430_processed.csv'
    m = re.search(r"system_(\d+)", path.name)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not infer site_id for file {path.name}")


# ------------- MAIN BUILD LOGIC -------------

def build_unified_dataset():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {IN_DIR}")

    all_dfs = []

    print(f"Found {len(files)} site files in {IN_DIR}")

    for f in files:
        print(f"[LOAD] {f.name}")
        df = load_site_file(f)
        site_id = infer_site_id(f, df)
        df["site_id"] = site_id
        all_dfs.append(df)

    # Concatenate all sites
    combined = pd.concat(all_dfs, axis=0)
    combined = combined.sort_index()

    # Ensure no NaNs in core features (should already be clean)
    if combined[FEATURE_COLS].isna().any().any():
        print("[WARN] NaNs detected in core features after concatenation.")

        # Compute per-site normalization statistics
    grouped_all = combined.groupby("site_id")
    means = grouped_all[FEATURE_COLS].mean()
    stds = grouped_all[FEATURE_COLS].std(ddof=0)  # population std

    # Avoid division by zero: replace 0 std with 1
    stds = stds.replace(0.0, 1.0)

    # Save stats for future use / reproducibility
    stats = pd.concat(
        {
            "mean": means,
            "std": stds,
        },
        axis=1
    )
    stats.to_csv(STATS_PATH)
    print(f"[OK] Wrote per-site normalization stats to {STATS_PATH}")

    # Apply per-site normalization using groupby.transform
    for col in FEATURE_COLS:
        norm_col_name = f"{col}_norm"
        combined[norm_col_name] = combined.groupby("site_id")[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1.0)
        )

    # Final unified dataset
    UNIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(UNIFIED_PATH)
    print(f"[OK] Wrote unified dataset to {UNIFIED_PATH}")


if __name__ == "__main__":
    build_unified_dataset()
