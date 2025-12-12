import re
import pandas as pd
from pathlib import Path

CONCAT_DIR = Path("data/raw/concatenated")
META_PATH = Path("data/systemdata/system_data.csv")

# UPDATE THIS to match your CSV's actual DC capacity column name:
DC_CAP_COL = "dc_capacity_kW"   # <-- CHANGE IF NEEDED


# Broad regex patterns
POWER_PATTERNS = re.compile(
    r"(ac[_ ]?power|real[_ ]?power|power[_ ]?kw|kw$|dc[_ ]?power|inv.*power)",
    re.IGNORECASE,
)

IRRADIANCE_PATTERNS = re.compile(
    r"(poa|irradiance|ghi)",
    re.IGNORECASE,
)

TEMP_PATTERNS = re.compile(
    r"(module[_ ]?temp|mod[_ ]?temp|cell[_ ]?temp|ambient[_ ]?temp|temp)",
    re.IGNORECASE,
)


def classify_columns(cols):
    has_power = any(bool(POWER_PATTERNS.search(c)) for c in cols)
    has_irr = any(bool(IRRADIANCE_PATTERNS.search(c)) for c in cols)
    has_temp = any(bool(TEMP_PATTERNS.search(c)) for c in cols)

    return has_power, has_irr, has_temp


def main():
    # Load metadata
    meta = pd.read_csv(META_PATH)
    if DC_CAP_COL not in meta.columns:
        raise ValueError(
            f"DC capacity column '{DC_CAP_COL}' not found in metadata. "
            f"Available columns: {list(meta.columns)}"
        )

    results = []
    for csv_file in CONCAT_DIR.glob("system_*_data.csv"):
        system_id = int(csv_file.stem.split("_")[1])

        # Only load headers
        df = pd.read_csv(csv_file, nrows=5)
        cols = df.columns

        has_power, has_irr, has_temp = classify_columns(cols)
        usable = has_power and has_irr and has_temp

        # Get capacity if present; otherwise NaN
        cap = meta.loc[meta["system_id"] == system_id, DC_CAP_COL]
        cap = cap.iloc[0] if len(cap) > 0 else float("nan")

        results.append({
            "system_id": system_id,
            "has_power": has_power,
            "has_irradiance": has_irr,
            "has_temperature": has_temp,
            "usable": usable,
            DC_CAP_COL: cap
        })

    out = pd.DataFrame(results).sort_values("system_id")
    out.to_csv("data/interim/feature_presence.csv", index=False)

    print(out)

    # Print usable sites + capacity (newline-separated)
    print("\n=== Usable Sites + DC Capacity ===\n")
    usable_df = out[out["usable"] == True]

    for _, row in usable_df.iterrows():
        print(f"{row['system_id']} — {row[DC_CAP_COL]}")
    print()


if __name__ == "__main__":
    main()
