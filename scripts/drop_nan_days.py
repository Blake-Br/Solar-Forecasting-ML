import pandas as pd
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parents[1]
IN_DIR = BASE_DIR / "data" / "processed_clean"
OUT_DIR = BASE_DIR / "data" / "processed_final"

# Columns that must be NaN-free
KEY_COLS = ["ac_power_kw", "poa_irradiance", "module_temp"]


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a DatetimeIndex using a known time column.
    Tries 'measured_on' first, then 'timestamp'.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    time_col = None
    for col in ["measured_on", "timestamp"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        raise ValueError("No 'measured_on' or 'timestamp' column found.")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    return df


def drop_days_with_any_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop entire days where ANY NaN appears in any KEY_COLS.
    """
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Boolean: row has any NaN in KEY_COLS
    row_has_nan = df[KEY_COLS].isna().any(axis=1)

    # Map datetime index to date
    dates = df.index.date

    # For each date, check if any row that day has NaN
    date_has_nan = pd.Series(row_has_nan.values, index=dates).groupby(level=0).any()

    # Dates to drop
    bad_dates = set(date_has_nan[date_has_nan].index)

    if not bad_dates:
        return df

    # Build mask: keep rows whose date is NOT in bad_dates
    keep_mask = ~pd.Index(df.index.date).isin(bad_dates)
    return df.loc[keep_mask]


def process_file(path_in: Path, path_out: Path):
    print(f"[LOAD] {path_in.name}")
    df = pd.read_csv(path_in)

    # Ensure datetime index
    df = ensure_datetime_index(df)

    # Drop days with any NaN in key columns
    before = len(df)
    df_clean = drop_days_with_any_nan(df)
    after = len(df_clean)

    dropped = before - after
    print(f"  Rows before: {before}, after: {after}, dropped: {dropped}")

    # Final sanity: drop any completely empty rows in key cols (shouldn't remain)
    df_clean = df_clean.dropna(subset=KEY_COLS, how="any")

    # Write out
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(path_out, index=True)
    print(f"[OK] Wrote cleaned file: {path_out.name}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {IN_DIR}")
        return

    for f in files:
        out_path = OUT_DIR / f.name
        try:
            process_file(f, out_path)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}\n")


if __name__ == "__main__":
    main()
