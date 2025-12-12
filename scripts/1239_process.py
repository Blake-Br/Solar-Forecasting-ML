import pandas as pd

df = pd.read_csv("data/processed/system_1239_data.csv")

df["measured_on"] = pd.to_datetime(df["measured_on"])

cutoff = pd.Timestamp("8/4/2018 15:15")

mask = df["measured_on"] >= cutoff

# AC: W → kW
df.loc[mask, "ac_power_kw"] = df.loc[mask, "ac_power_kw"] / 1000.0

# Temp: °C → °F
df.loc[mask, "module_temp"] = df.loc[mask, "module_temp"] * 9/5 + 32

df.to_csv("data/processed/system_1239_data.csv", index=False)
