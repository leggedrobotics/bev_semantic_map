import argparse
from pathlib import Path
import h5py
from prettytable import PrettyTable
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="/Data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2",
    )
    args = parser.parse_args()

    h5py_files = [str(s) for s in Path(args.target).rglob("*.h5py") if str(s).find("ignored_bags") == -1]
    h5py_files.sort()
    print(h5py_files)
    for h5py_file in h5py_files:
        with h5py.File(h5py_file, "r+") as file:
            x = PrettyTable()
            x.field_names = ["Seq", "Key", "Data", "Elements"]
            for seq_name, seq in file.items():
                for k, v in seq.items():
                    try:
                        x.add_row([seq_name, k, np.array(v).shape[0], v["header_seq"].shape[0]])
                    except:
                        x.add_row([seq_name, k, np.array(v).shape[0], 1])
            print(x)
