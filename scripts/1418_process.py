import pandas as pd


def drop_zero_power_days_with_sun(
    df: pd.DataFrame,
    power_col: str = "ac_power_kw",
    irr_col: str = "poa_irradiance",
    day_start_hour: int = 8,
    day_end_hour: int = 16,
    irr_threshold: float = 50.0,
) -> pd.DataFrame:
    """
    Drop whole days where, during [day_start_hour, day_end_hour),
    AC power is zero for all intervals but irradiance is non-zero at least once.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Restrict to daytime window
    hours = df.index.hour
    daytime_mask = (hours >= day_start_hour) & (hours < day_end_hour)
    df_day = df.loc[daytime_mask].copy()

    # Group by date
    dates = df_day.index.date
    grouped = df_day.groupby(dates)

    bad_dates = []

    for date, grp in grouped:
        # Any irradiance above threshold?
        has_sun = (grp[irr_col] > irr_threshold).any()

        # Any non-zero power?
        has_power = (grp[power_col] > 0).any()

        # If there is sun but no power at all → bad day
        if has_sun and not has_power:
            bad_dates.append(date)

    # Convert bad_dates to a date index mask on the full df
    if not bad_dates:
        return df  # nothing to drop

    bad_dates = set(bad_dates)
    full_dates = df.index.date
    bad_mask = pd.Index(full_dates).isin(bad_dates)

    # Drop those days from the original df
    cleaned = df.loc[~bad_mask].copy()
    return cleaned

df = pd.read_csv("data/processed/system_1418_data.csv")

df["measured_on"] = pd.to_datetime(df["measured_on"])
df["ac_power_kw"] = df["ac_power_kw"] / 1000.0

df = df.set_index("measured_on").sort_index()

df_15 = df.resample("15min").mean()

df_15 = df_15.dropna(how="all")

df_clean = drop_zero_power_days_with_sun(df_15)

df_clean.to_csv("data/processed/system_1418_data.csv", index=True)