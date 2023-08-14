#!/usr/bin/env python

"""
Saves the x data from a provided graph as a new torch file.

Author: Robin Schmid
Date: Oct 2022
"""

import os
import glob
import torch
from tqdm import tqdm


GRAPH_PATH = "/home/rschmid/RosBags/perugia_fixed/features/all_feat"

if __name__ == "__main__":
    print("Saving features from graph")

    file_names = [os.path.basename(d) for d in sorted(glob.glob(GRAPH_PATH + "/*"))]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for file in tqdm(file_names):
        graph = torch.load(os.path.join(GRAPH_PATH, file), map_location=device)
        # # Save only the x value of the graph
        torch.save(graph.x, os.path.join(GRAPH_PATH, file))
