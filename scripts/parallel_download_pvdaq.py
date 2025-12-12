from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
from pathlib import Path

import pandas as pd
import pvdaq_access
from dotenv import load_dotenv

# =======================
# CONFIG
# =======================

# Project root = parent of this script's directory
BASE_DIR = Path(__file__).resolve().parents[1]

META_PATH = BASE_DIR / "data" / "systemdata" / "system_data.csv"
RAW_ROOT = BASE_DIR / "data" / "raw"
CONCAT_ROOT = RAW_ROOT / "concatenated"

# Change this to match the size column in your CSV
SIZE_COL = "dataset_size_mb"      # e.g. "size_mb" / "Size (MB)" / etc.
MIN_MB = 20.0
MAX_MB = 100.0

MAX_WORKERS = 4  # number of parallel downloads


# =======================
# ENV + PATH INIT
# =======================

def init_env():
    """Load API key and ensure directories exist."""
    load_dotenv(BASE_DIR / "configs" / ".env")
    api_key = os.getenv("NREL_API_KEY")
    if not api_key:
        raise RuntimeError("NREL_API_KEY not set in configs/.env or environment.")

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    CONCAT_ROOT.mkdir(parents=True, exist_ok=True)


# =======================
# PER-SYSTEM WORKER
# =======================

def process_system(system_id: int):
    """
    Download and concatenate data for a single PVDAQ system.
    Idempotent: if final concatenated file exists, it skips.
    """
    system_id = int(system_id)  # keep as int for naming, logging

    final_name = f"system_{system_id}_data.csv"
    final_path = CONCAT_ROOT / final_name

    # Skip if already done
    if final_path.exists():
        print(f"[SKIP] system {system_id}: already downloaded at {final_path}")
        return

    raw_dir = RAW_ROOT / f"system_{system_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[START] system {system_id}: downloading to {raw_dir}")

    try:
        # NOTE: cast system_id to str for pvdaq_access calls
        sid_str = str(system_id)

        # Download all raw files for this system
        pvdaq_access.downloadData(sid_str, str(raw_dir), file_type="csv")

        # Concatenate within raw_dir: writes system_<id>_data.csv into raw_dir
        pvdaq_access.concatenateData(sid_str, str(raw_dir))

        temp_concat = raw_dir / final_name
        if not temp_concat.exists():
            raise FileNotFoundError(
                f"Expected concatenated file not found: {temp_concat}"
            )

        # Move concatenated file to CONCAT_ROOT
        shutil.move(str(temp_concat), str(final_path))
        print(f"[OK] system {system_id}: concatenated file at {final_path}")

    except Exception as e:
        print(f"[FAIL] system {system_id}: {e}")
    finally:
        # Optional cleanup of per-day files
        if raw_dir.exists():
            for f in raw_dir.iterdir():
                if f.is_file():
                    f.unlink()
            try:
                raw_dir.rmdir()
            except OSError:
                pass


# =======================
# SITE SELECTION
# =======================

def select_sites_by_size():
    """
    Read metadata CSV, filter by size, return DataFrame of selected sites.
    """
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {META_PATH}")

    df = pd.read_csv(META_PATH)
    df[SIZE_COL] = pd.to_numeric(df[SIZE_COL], errors="coerce")


    if SIZE_COL not in df.columns:
        raise ValueError(
            f"Column '{SIZE_COL}' not found in {META_PATH}. "
            f"Columns: {list(df.columns)}"
        )
    if "system_id" not in df.columns:
        raise ValueError("Metadata must contain a 'dataset_system_id' column.")

    mask = (df[SIZE_COL] >= MIN_MB) & (df[SIZE_COL] <= MAX_MB)
    selected = df.loc[mask].copy()

    selected = selected.sort_values(SIZE_COL, ascending=False).reset_index(drop=True)

    print(
        f"Selected {len(selected)} systems between {MIN_MB} and {MAX_MB} MB "
        f"(sorted by {SIZE_COL})."
    )
    return selected


# =======================
# PARALLEL ORCHESTRATOR
# =======================

def run_parallel_download(system_ids):
    """
    Run process_system(system_id) in parallel for a list of system_ids,
    using a bounded thread pool.
    """
    print(f"Starting parallel download for {len(system_ids)} systems "
          f"with max_workers={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_system, int(sid)): int(sid)
            for sid in system_ids
        }

        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] system {sid} task failed: {e}")


# =======================
# MAIN
# =======================

if __name__ == "__main__":
    init_env()

    selected = select_sites_by_size()
    system_ids = selected["system_id"].tolist()

    if not system_ids:
        print("No systems found in the specified size range. Exiting.")
    else:
        run_parallel_download(system_ids)
