import csv
import os
from pathlib import Path
from data_processing import process_system as process_system
from data_processing import open_data as open_data
from data_processing import get_chars as get_chars

def load_valid_system_ids():
    system_file = Path("data/systemdata/system_data.csv")
    ids = set()

    with system_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                ids.add(row[0].strip())
                
    return ids


def get_system_id() -> str:
    while True:
        system_id = input("Enter system ID (q to go back): ").strip()
        if system_id == 'q': return -1
        VALID_SYSTEM_IDS = load_valid_system_ids()
        if not system_id:
            print("Error: System ID cannot be empty")
            continue
        if system_id not in VALID_SYSTEM_IDS:
            print("Error: System ID {system_id} not found in system_data.csv")
            return 0
        return system_id

def main():
    while True:
        print("\n=== Data Hub ===")
        print("1 -- Get system data")
        print("2 -- Remove system data")
        print("3 -- Open data")
        print("4 -- Data characteristics")
        
        print("q -- Quit")

        choice = input("Select an option: ").strip().lower()

        match choice:
            case "1":
                system_id = get_system_id()
                if system_id == -1: continue
                target = Path(f"data/processed/system_{system_id}_data.csv")
                if target.exists():
                    print("Error: system data already exists")
                else: process_system(system_id)
            
            case "2":
                system_id = get_system_id()
                if system_id == -1: continue
                target = Path(f"data/processed/system_{system_id}_data.csv")
                if not target.exists():
                    print("Error: system data does not exist exists")
                else: os.remove(target)
            case "3":
                system_id = get_system_id()
                if system_id == -1: continue
                open_data(system_id)
            case "4":
                system_id = get_system_id()
                if system_id == -1: continue
                get_chars(system_id)
            
            case "q":
                break

if __name__ == "__main__":
    main()