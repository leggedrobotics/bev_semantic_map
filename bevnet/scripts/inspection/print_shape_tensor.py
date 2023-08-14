#!/usr/bin/env python

"""
Prints the shape of torch tensor or numpy array for inspection.

Author: Robin Schmid
Date: Oct 2022
"""

import torch
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-path", type=str, required=True, help="""Path to the torch tensor"""
    )
    parser.add_argument(
        "--seg", action='store_true'
    )
    args = parser.parse_args()

    tensor = torch.load(args.data_path, map_location="cpu")
    # tensor = np.load(args.data_path)

    if args.seg:
        print(torch.unique(tensor))

    try:
        # For torch tensors
        print("Shape:", tensor.shape)
        print("Type:", tensor.dtype)
        print("Mean + Var:", torch.var_mean(tensor))

    except:
        # For numpy arrays or lists
        print("Len 1:", len(tensor))
        print("Len 2:", len(tensor[0]))
        print("Mean + Var:", np.asarray(tensor).mean(), np.asarray(tensor).var())

    finally:
        pass
