import os
import argparse
from pathlib import Path
import pvdaq_access
from dotenv import load_dotenv
import shutil

def process_system(system: str):
    # Load API key
    load_dotenv("configs/.env")
    assert os.getenv("NREL_API_KEY"), "Set NREL_API_KEY first"

    
    base_dir = Path("data")  # main data folder
    raw_dir = base_dir / "raw" / f"system_{system}"
    proc_dir = base_dir / "processed"

    # Create system-specific subfolder
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    final_name = f"system_{system}_data.csv"
    final_file = raw_dir / final_name

    file_type = "csv"

    # Download into that subfolder
    pvdaq_access.downloadData(system, str(raw_dir), file_type=file_type)
    pvdaq_access.concatenateData(system, str(raw_dir))

    # move concatenated data, remove individual csv files
    shutil.move(str(final_file), str(proc_dir / final_name))

    for f in raw_dir.iterdir():
        if f.is_file():
            f.unlink()
    raw_dir.rmdir()

def open_data(system: str):
    target = Path(f"data/processed/system_{system}_data.csv")
    if not target.exists():
        print("Error: system data has not been downloaded. Go download it.")
        return
    os.startfile(target) 
    print(f"Opening {target}")