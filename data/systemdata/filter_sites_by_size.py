import pandas as pd

def filter_sites_by_size(
    csv_path: str,
    size_mb_column: str = "dataset_size_mb",
    min_mb: float = 20.0,
    max_mb: float = 100.0,
):
    """
    Reads metadata and returns system IDs whose dataset size
    is between min_mb and max_mb (inclusive).
    """
    df = pd.read_csv(csv_path)

    # Check required columns
    if size_mb_column not in df.columns:
        raise ValueError(
            f"Column '{size_mb_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Filter based on size bounds
    mask = (df[size_mb_column] >= min_mb) & (df[size_mb_column] <= max_mb)
    selected = df.loc[mask]

    print(f"Found {len(selected)} systems between {min_mb}–{max_mb} MB:\n")
    print(selected)

    # Return a list of system IDs
    if "system_id" in selected.columns:
        return selected["system_id"].tolist()
    else:
        raise ValueError("CSV must contain a 'system_id' column.")


if __name__ == "__main__":
    systems = filter_sites_by_size("data/systemdata/system_data.csv")
    print("\nSystem IDs:")
    print(systems)
