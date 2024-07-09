import argparse
import h5py
from perception_bev_learning.dataset.bev_dataset import get_sequence_key

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--h5_file",
        type=str,
        help="h5 file to be read",
        default="/data/bev_traversability/tensor01/racer-jpl9_hw_2023-08-29-21-48-16-UTC_halter-ranch-dubost-2_ROBOT_dubost-datacollect-patrick-1.h5py",
    )

    args = vars(parser.parse_args())

    if args["h5_file"] == "nan":
        print("Please provide the path to the h5 file to be read")
        exit()
    else:
        h5py_file = args["h5_file"]

    with h5py.File(h5py_file, "r") as hdf_file:
        # Print the keys at the root level of the HDF5 file
        print("Keys in the root of the HDF5 file:")
        h5_keys = list(hdf_file.keys())
        print(h5_keys)
        for i in h5_keys:
            print(list(hdf_file[i].keys()))
            print(hdf_file[i]["crl_rzr_traversability_map_micro_debug_map"].keys())

        # # You can explore specific datasets or groups within the HDF5 file using their keys
        # # For example, if there's a dataset named 'data' in the root of the file:
        # if 'data' in hdf_file['bev']['map_micro_debug']:
        #     dataset = hdf_file['bev']['map_micro_debug']['data']
        #     print("Shape of the 'data' dataset:", dataset.shape)
        #     print("Data type of the 'data' dataset:", dataset.dtype)
        #     print("Data in the 'data' dataset:")
        #     print(dataset[:])  # Print the entire dataset or use slicing to print specific portions

        # You can similarly explore other datasets or groups within the file using their keys
        # For nested groups or datasets, you can navigate through the hierarchy using dictionary-like syntax.
