import os
import copy
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import argparse

import rosbag
import time

from racerenv import cleanup


def run_stack(data_dir, mode, env_file, visualization=False):
    # """Run a mission test"""
    env = copy.deepcopy(os.environ)
    env.update(
        {
            "ROSBAG_DIR": str(data_dir),
        }
    )
    # Clean up previous session
    cleanup.main(None)

    # Launch rosmaster
    roscore_process = subprocess.Popen(
        ["roscore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
    )
    # make sure everything is up before moving on
    time.sleep(10)
    print(f"starting {data_dir}")
    # Run simulation
    try:
        subprocess.call(
            [
                "racerenv",
                "run-replay",
                "-c",
                f"{mode}",
                f"-d",
                f"{data_dir}",
                f"--env_file_override",
                f"{env_file}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except:
        print("error")

    time.sleep(10)
    # Clean up
    cleanup.main(None)
    roscore_process.kill()


if __name__ == "__main__":
    """
    Provide a directory to a single trajectory using the -b or pass a folder containing multiple trajectories.
    Script will start the racer_replay_bev_semantics and racer_replay_bev_trav tmux session.
    """

    parser = argparse.ArgumentParser()
    # args used in all types
    parser.add_argument(
        "-b",
        "--bag",
        type=str,
        help="Path to a folder directly containing the bag files of a single trajectory",
        default="/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2/jpl6_camp_roberts_shakeout_y6_d2_t6_Tue_Jun__7_23-29-08_2022_utc",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Path to a folder containing multiple trajectories",
        default="/data/bev_traversability/2022-06-07-jpl6_camp_roberts_d2",
    )

    env_file = "/data_bev/bev_traversability/env.txt"

    args = vars(parser.parse_args())
    result_folders = [args["bag"]]
    if args["folder"] != "nan":
        result_folders = [
            os.path.join(args["folder"], o)
            for o in os.listdir(args["folder"])
            if os.path.isdir(os.path.join(args["folder"], o))
        ]
        result_folders = result_folders
        result_folders.sort()

    for f in result_folders:
        print("Going to process: ", f)

    for counter, folder in enumerate(result_folders):
        if (
            len(
                [
                    str(s)
                    for s in Path(folder).rglob("*.bag")
                    if str(s).find("crl_rzr_bev_trav_") != -1
                ]
            )
            > 0
        ):
            print("Skipping folder:", folder)
            print(f"processing ({counter}/{len(result_folders)})")
            continue

        print(folder)
        print(f"processing ({counter}/{len(result_folders)})")

        global_free_mem, total_mem = torch.cuda.mem_get_info()

        print("Free memory: ", global_free_mem / 1e9)
        # if global_free_mem / 1e9 > 11:
        # More than 15 GB-VRAM run everything together
        # input("Press Enter to continue...")
        run_stack(folder, "racer_replay_bev_full.yaml", env_file)
        # run_stack(folder, "racer_replay_bev_semantics.yaml")
        # run_stack(folder, "racer_replay_bev_trav.yaml")

        # elif global_free_mem / 1e9 > 8:
        #     # Run stack seperate
        #     run_stack(folder, "racer_replay_bev_semantics.yaml")
        #     run_stack(folder, "racer_replay_bev_trav.yaml")
        # else:
        #     print("Most likely amount of VRAM is insufficient to run the stack.")
