import pandas as pd
from pathlib import Path

BASE = Path("data/processed_final")

SIGNIFICANT_COLS = [
    "ac_power_kw",
    "poa_irradiance",
    "module_temp",
]

def main():
    files = sorted(BASE.glob("*.csv"))
    if not files:
        print("No files in processed_clean/")
        return

    print(f"Checking {len(files)} files...\n")

    for f in files:
        df = pd.read_csv(f)

        missing_cols = [c for c in SIGNIFICANT_COLS if c not in df.columns]
        if missing_cols:
            print(f"[ERROR] {f.name}: Missing expected columns: {missing_cols}")
            continue

        # count NaNs per column
        nan_counts = df[SIGNIFICANT_COLS].isna().sum()

        # any NaNs at all?
        total_nans = nan_counts.sum()

        if total_nans == 0:
            print(f"[OK] {f.name}: No NaNs in significant columns.")
        else:
            print(f"[WARN] {f.name}: {total_nans} total NaNs")
            print(nan_counts)
            print()

if __name__ == "__main__":
    main()
