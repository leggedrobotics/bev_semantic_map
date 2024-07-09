import argparse
from pathlib import Path


def get_h5py_files(default_file="nan", default_folder="nan", search_key="*with_data.h5py"):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fi",
        "--file",
        type=str,
        default=default_file,
    )
    parser.add_argument(
        "-fo",
        "--folder",
        type=str,
        default=default_folder,  # "nan",
    )
    args = parser.parse_args()
    if args.folder != "nan":
        h5py_files = [str(s) for s in Path(args.folder).glob(search_key)]
        h5py_files.sort()
    else:
        h5py_files = [args.file]

    return h5py_files
