import pandas as pd

# --- EDIT THIS ---
FILE_PATH = "data/processed_clean/system_1418_data.csv"
OUT_PATH  = "data/processed_clean/system_1418_data.csv"
# -----------------

df = pd.read_csv(FILE_PATH)

if "ac_power_kw" not in df.columns:
    raise ValueError("Column 'ac_power_kw' not found in file.")

print("NaNs before:", df["ac_power_kw"].isna().sum())

# Replace NaN with 0 ONLY for ac_power_kw
df["ac_power_kw"] = df["ac_power_kw"].fillna(0)

print("NaNs after:", df["ac_power_kw"].isna().sum())

# Save the result
df.to_csv(OUT_PATH, index=False)
print(f"Saved fixed file to: {OUT_PATH}")
