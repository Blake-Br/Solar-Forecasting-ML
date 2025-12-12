import pandas as pd
from pathlib import Path

# -------- CONFIG --------

BASE_DIR = Path(__file__).resolve().parents[1]
IN_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "data" / "processed_clean"

# core feature columns expected in processed files
FEATURE_COLS = ["ac_power_kw", "poa_irradiance", "module_temp"]

# daytime window and irradiance threshold ("sunny" definition)
DAY_START_HOUR = 8
DAY_END_HOUR = 16
IRR_THRESHOLD = 100.0  # W/m^2

# max length (in consecutive intervals) to interpolate AC power gaps
INTERP_LIMIT = 2  # e.g. up to 30 min if 15-min data


# -------- HELPERS --------

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a DatetimeIndex.
    - If 'timestamp' column exists, use it.
    - Else if index is already DatetimeIndex, leave it.
    - Else raise.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    # try common column name
    for col in ["timestamp", "measured_on", "date", "time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            return df.sort_index()

    raise ValueError("No suitable timestamp column/index found for datetime index.")


def drop_empty_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where all core feature columns are NaN.
    These are rows that only have timestamp + system_id (or similar).
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    return df.dropna(subset=FEATURE_COLS, how="all")


def drop_zero_power_days_with_sun(
    df: pd.DataFrame,
    power_col: str = "ac_power_kw",
    irr_col: str = "poa_irradiance",
    day_start_hour: int = DAY_START_HOUR,
    day_end_hour: int = DAY_END_HOUR,
    irr_threshold: float = IRR_THRESHOLD,
) -> pd.DataFrame:
    """
    Drop entire days where:
      - during [day_start_hour, day_end_hour),
      - irradiance > irr_threshold at least once (sunny),
      - but power_col is 0 for all those sunny intervals.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    hours = df.index.hour
    day_mask = (hours >= day_start_hour) & (hours < day_end_hour)
    df_day = df.loc[day_mask].copy()

    bad_dates = []
    for date, grp in df_day.groupby(df_day.index.date):
        sun_mask = grp[irr_col] > irr_threshold
        if not sun_mask.any():
            continue  # not a sunny day, ignore

        has_power = (grp.loc[sun_mask, power_col] > 0).any()
        if not has_power:
            bad_dates.append(date)

    if not bad_dates:
        return df

    bad_dates = set(bad_dates)
    keep_mask = ~pd.Index(df.index.date).isin(bad_dates)
    return df.loc[keep_mask]


def drop_days_with_nan_power_in_sun(
    df: pd.DataFrame,
    power_col: str = "ac_power_kw",
    irr_col: str = "poa_irradiance",
    day_start_hour: int = DAY_START_HOUR,
    day_end_hour: int = DAY_END_HOUR,
    irr_threshold: float = IRR_THRESHOLD,
    nan_frac_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Drop days where, during [day_start_hour, day_end_hour),
    irradiance indicates sun but power_col is NaN for more than
    nan_frac_threshold fraction of those sunny intervals.
    """
    hours = df.index.hour
    day_mask = (hours >= day_start_hour) & (hours < day_end_hour)
    df_day = df.loc[day_mask].copy()

    bad_dates = []
    for date, grp in df_day.groupby(df_day.index.date):
        sun_mask = grp[irr_col] > irr_threshold
        if not sun_mask.any():
            continue

        sun_grp = grp.loc[sun_mask]
        if sun_grp.empty:
            continue

        nan_frac = sun_grp[power_col].isna().mean()
        if nan_frac > nan_frac_threshold:
            bad_dates.append(date)

    if not bad_dates:
        return df

    bad_dates = set(bad_dates)
    keep_mask = ~pd.Index(df.index.date).isin(bad_dates)
    return df.loc[keep_mask]


def clean_site_file(path_in: Path, path_out: Path):
    print(f"[LOAD] {path_in}")
    df = pd.read_csv(path_in)

    # ensure datetime index
    df = ensure_datetime_index(df)

    # drop rows with only timestamp + system_id (no features)
    df = drop_empty_feature_rows(df)

    # interpolate short AC power gaps
    df["ac_power_kw"] = df["ac_power_kw"].interpolate(limit=INTERP_LIMIT)

    # drop days with sun but zero power all day
    df = drop_zero_power_days_with_sun(df)

    # drop days with sun but AC power NaN for most of the day
    df = drop_days_with_nan_power_in_sun(df)

    # final safety: drop any remaining rows with all features missing
    df = df.dropna(subset=FEATURE_COLS, how="all")

    # write out
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_out, index=True)
    print(f"[OK] wrote cleaned file to {path_out}\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # process all csv files in data/processed
    for path_in in sorted(IN_DIR.glob("*.csv")):
        # example: system_1430_processed.csv -> same name in processed_clean
        path_out = OUT_DIR / path_in.name
        try:
            clean_site_file(path_in, path_out)
        except Exception as e:
            print(f"[ERROR] {path_in}: {e}")


if __name__ == "__main__":
    main()
