import os
import pandas as pd
import argparse
import numpy as np
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

def get_chars(system: str):
    target = Path(f"data/processed/system_{system}_data.csv")
    if not target.exists():    
        print("Error: system data has not been downloaded. Go download it.")
        return
    df = pd.read_csv(target)
    print("Attributes: ")
    for c in df.columns:
        print(c)

def change_col(system: str, attr_old: str, attr_new: str):
    target = Path(f"data/processed/system_{system}_data.csv")
    if not target.exists():    
        print("Error: system data has not been downloaded. Go download it.")
        return
    df = pd.read_csv(target)
    if attr_old in df.columns:
        df.rename(columns={target: attr_new})
        df.to_csv(target, index=False)
    else:
        print("Attribute not found in csv file.")

def remove_col(system: str, attr_to_rem: str):
    target = Path(f"data/processed/system_{system}_data.csv")
    if not target.exists():    
        print("Error: system data has not been downloaded. Go download it.")
        return
    df = pd.read_csv(target)
    if attr_to_rem in df.columns:
        df.drop(columns=[attr_to_rem])
        df.to_csv(target, index=False)
    else:
        print("Attribute not found in csv file.")

def add_col(system: str, attr_to_add: str, vals=None):
    target = Path(f"data/processed/system_{system}_data.csv")
    if not target.exists():    
        print("Error: system data has not been downloaded. Go download it.")
        return
    df = pd.read_csv(target)
    if attr_to_add in df.columns:
        print("Attribute already in csv file.")
        return
    
    if vals is None:
        vals = np.zeros(len(df))

    if len(vals) != len(df):
        print("New attribute rows do not match data")
        return

    df.insert(len(df.columns), attr_to_add, vals)

    df.to_csv(target, index=False)
    