from pathlib import Path
import csv

script_dir = Path(__file__).parent
parent_dir = script_dir.parent 

# 1. Define the base 'raw' directory
raw_dir = parent_dir / 'data' / 'raw'

# 2. Use .glob('*') to find all items (directories) inside 'raw'
for system_dir in raw_dir.glob('*'):
    
    # Make sure it's a directory
    if not system_dir.is_dir():
        continue

    # 3. Get the directory name (e.g., "system_1257")
    system_id = system_dir.name 
    
    # 4. Construct the exact filename you expect
    file_name = f"{system_id}_data.csv"
    file_path = system_dir / file_name

    # 5. Check if that specific file exists before trying to open it
    if file_path.is_file():
        try:
            # Use .relative_to() for a cleaner print message
            print(f"--- Processing: {file_path.relative_to(parent_dir)} ---")
            
            with open(file_path, mode='r') as f:
                reader = csv.reader(f)
                column_names = next(reader)
            
            print(column_names)
            print("") # Add a blank line for readability

        except StopIteration:
            # This handles files that are completely empty
            print(f"File is empty: {file_path.name}\n")
        except Exception as e:
            # Catch any other reading errors
            print(f"Could not read {file_path.name}: {e}\n")
    else:
        # Optional: let you know if a directory didn't contain the expected file
        print(f"--- Skipping: Did not find '{file_name}' in {system_dir.name} ---\n")