import pickle
import pandas as pd
from pathlib import Path
import os

cfg_dir = "/data_large/manthan/camp_roberts_f/config"


# List of pickle files in provided dir
pickle_files = [str(s) for s in Path(cfg_dir).rglob("*.pkl")]

print(pickle_files)


# Load the data from the pickle file
with open(pickle_files[0], "rb") as file:
    data = pickle.load(file)

for i in range(1, len(pickle_files)):
    with open(pickle_files[i], "rb") as file:
        curr_data = pickle.load(file)
        data.extend(curr_data)

pkl_cfg_file = os.path.join(cfg_dir, "dataset_config.pkl")
# Save to dataset_config pickle file
print(f"Saving config to {pkl_cfg_file}")

with open(pkl_cfg_file, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
