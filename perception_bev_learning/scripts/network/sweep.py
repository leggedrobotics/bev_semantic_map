from perception_bev_learning.utils import load_yaml
from perception_bev_learning import BEV_ROOT_DIR
from argparse import ArgumentParser
import os

import copy
import os
import shlex
import shutil
import subprocess
import time

if __name__ == "__main__":
    """
    Performs full multiple training runs.
    Takes as an input a sweep-configuration yaml-file.
    The file can define the following:

    sweep_dir: sweep1         # Name of the sweep directory

    common:                   # Parameters applied to all runs
        max_steps: 16000
        ..
    runs:
        lss_only:             # RUN1 Defines a run with name lss_only
          network: skip       # Command Line paramter of the run
          ..
        pointcloud_only:      # RUN2 ...
          network: pointcloud # ...
          ..
    """
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", default="sweep.yaml")
    args = parser.parse_args()

    config = load_yaml(os.path.join(BEV_ROOT_DIR, "cfg/exp", args.file))

    print(config)
    env = os.environ["ENV_WORKSTATION_NAME"]
    cmd = f"{BEV_ROOT_DIR}/scripts/ws/job_{env}.sh python {BEV_ROOT_DIR}/scripts/network/train.py --help"
    my_env = copy.copy(os.environ)
    # Support both list and str format for commands
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    p = subprocess.Popen(cmd, cwd=None, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    ls = str(out).split("--")[2:]
    ls = [s[: s.find(" ")] for s in ls]

    for name, run in config["runs"].items():
        for arg_name, arg_value in run.items():
            if not (arg_name in ls):
                raise ValueError(f"Invalid argument {arg_name}")

    print("All configuration names are valid!")

    for name, run in config["runs"].items():
        sweep_dir = config["sweep_dir"]
        cmd_args = f"--general.name={sweep_dir}/{name} "

        run.update(config.get("common", {}))

        for arg_name, arg_value in run.items():
            cmd_args += f"--{arg_name}={arg_value} "

        cmd = f"{BEV_ROOT_DIR}/scripts/ws/job_{env}.sh python {BEV_ROOT_DIR}/scripts/network/train.py " + cmd_args
        print(cmd)
        os.system(cmd)
