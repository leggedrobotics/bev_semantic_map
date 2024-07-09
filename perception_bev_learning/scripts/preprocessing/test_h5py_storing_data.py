from perception_bev_learning.utils import DatasetWriter
import numpy as np
import h5py
from pytictac import CpuTimer
import random

dataset_file = "/tmp/test.h5py"

sequence = "a"

fieldname = "rostopic_a"

for entry in [10, 100, 1000, 10000]:
    print(f"Start writing {entry}")
    dataset_writer = DatasetWriter(dataset_file)
    # writing random dataset
    data = {"static_a": np.random.rand(100, 100), "static_b": np.random.rand(3, 3)}

    for i in range(entry):
        dataset_writer.add_static(sequence, fieldname, data)

    dynamic_dict = {"dyn_a": np.random.rand(640, 480), "dyn_b": np.random.rand(3, 3)}

    for i in range(entry):
        dataset_writer.add_data(sequence, fieldname, dynamic_dict)

    dataset_writer.close()

    # reading data speed test
    handle = h5py.File(dataset_file, "r")
    with CpuTimer(f"dyna_a reading {entry}"):
        for i in range(1000):
            data = handle[sequence][fieldname]
            out = data["dyn_a"][random.randint(0, entry - 1)] + data["dyn_a"][random.randint(0, entry - 1)]

    with CpuTimer(f"dyna_b reading {entry}"):
        for i in range(1000):
            data = handle[sequence][fieldname]
            out = data["dyn_b"][random.randint(0, entry - 1)] + data["dyn_b"][random.randint(0, entry - 1)]
    handle.close()
