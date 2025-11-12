import os
from pathlib import Path
import pvdaq_access
from dotenv import load_dotenv

# Load API key
load_dotenv("configs/.env")
assert os.getenv("NREL_API_KEY"), "Set NREL_API_KEY first"

# Choose system ID
system = "1430"              # pick any valid system
base_dir = Path("data/raw")  # main raw folder

# Create system-specific subfolder
outdir = base_dir / f"system_{system}"
outdir.mkdir(parents=True, exist_ok=True)

file_type = "csv"

# Download into that subfolder
#pvdaq_access.downloadData(system, str(outdir), file_type=file_type)
pvdaq_access.concatenateData(system, str(outdir))

print(f"Done. Data saved to {outdir}")