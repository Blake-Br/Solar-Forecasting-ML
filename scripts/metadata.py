import pandas as pd
from pathlib import Path

METADATA_PATH = Path("data/systemdata/system_metadata.csv")

def load_metadata() -> pd.DataFrame:
    if METADATA_PATH.exists():
        return pd.read_csv(METADATA_PATH, dtype={"system_id": str})
    else:
        return pd.DataFrame(columns=[
            "system_id", "data_path", "file_size_bytes",
            "n_rows", "n_cols", "notes"
        ])

def save_metadata(df: pd.DataFrame) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)

def record_system(system_id: str, note: str = "") -> None:
    df_meta = load_metadata()

    data_path = Path(f"data/processed/system_{system_id}_data.csv")
    if not data_path.exists():
        print(f"Error: {data_path} does not exist.")
        return

    # get size and shape
    file_size = data_path.stat().st_size
    df_data = pd.read_csv(data_path)
    n_rows, n_cols = df_data.shape

    # remove old row for this system if it exists
    df_meta = df_meta[df_meta["system_id"] != system_id]

    new_row = {
        "system_id": system_id,
        "data_path": str(data_path),
        "file_size_bytes": file_size,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "notes": note,
    }

    df_meta = pd.concat([df_meta, pd.DataFrame([new_row])], ignore_index=True)
    save_metadata(df_meta)

def update_notes(system_id: str, note: str) -> None:
    df_meta = load_metadata()

    if system_id not in set(df_meta["system_id"]):
        print("Error: system not in metadata.")
        return

    df_meta.loc[df_meta["system_id"] == system_id, "notes"] = note
    save_metadata(df_meta)