import pickle
import pandas as pd

cfg_file = "/data_ssd/manthan/bev_traversability/camp_roberts/h5_gt/config/dataset_config_clean_seperation_subsample_1.pkl"
# Load the data from the pickle file
with open(cfg_file, "rb") as file:
    data = pickle.load(file)

# Assuming data is a list of dictionaries, you can convert it to a Pandas DataFrame
df_orig = pd.DataFrame(data)

# df_orig.to_csv("output.csv", index=False)
pd.set_option("display.max_rows", None)  # Show all rows

df = df_orig[
    df_orig["sequence_key"]
    != "jpl6_camp_roberts_shakeout_y1_d2_t1_C1_to_B2_Tue_Jun__7_18-21-54_2022_utc"
]
print(len(df))
df_train = df[df["mode"] == "train"]
df_val = df[df["mode"] == "val"]
df_test = df[df["mode"] == "test"]

# Print the DataFrame as a table
print(f"Training samples are: {len(df_train)}")
print(f"Validation samples are: {len(df_val)}")
print(f"Testing samples are: {len(df_test)}")

# old = "playback_slow"
# new = "racer-jpl9_hw_2023-08-29-21-48-16-UTC_halter-ranch-dubost-2_ROBOT_dubost-datacollect-patrick-1"
# df["sequence_key"] = new

# print(df)

# list_of_dicts = df.to_dict(orient="records")
# with open(cfg_file, "wb") as handle:
#     pickle.dump(list_of_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
