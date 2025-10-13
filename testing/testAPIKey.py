import os, pvdaq_access
from dotenv import load_dotenv
load_dotenv("configs/.env")

# ensure the key is in the environment
assert os.getenv("NREL_API_KEY"), "Set NREL_API_KEY first"

system = "1214"          # pick a valid system id
outdir = "data"          # will be created if missing
file_type = "csv"        # or "parquet"

pvdaq_access.downloadData(system, outdir, file_type=file_type)
print("done")
