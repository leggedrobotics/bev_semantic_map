import os
from pathlib import Path
import numpy as np
from fnmatch import fnmatchcase
import sys
from rosbag import Bag
from glob import glob
import subprocess


# Copied from https://www.clearpathrobotics.com/assets/downloads/support/merge_bag.py
def merge_bags_single(input_bag, output_bag, topics="*", verbose=False):
    topics = topics.split(" ")

    total_included_count = 0
    total_skipped_count = 0

    if verbose:
        print("Writing bag file: " + output_bag)
        print("Matching topics against patters: '%s'" % " ".join(topics))

    with Bag(output_bag, "w") as o:
        for ifile in input_bag:
            matchedtopics = []
            included_count = 0
            skipped_count = 0
            if verbose:
                print("> Reading bag file: " + ifile)
            with Bag(ifile, "r") as ib:
                for topic, msg, t in ib:
                    if any(fnmatchcase(topic, pattern) for pattern in topics):
                        if not topic in matchedtopics:
                            matchedtopics.append(topic)
                            if verbose:
                                print("Including matched topic '%s'" % topic)
                        o.write(topic, msg, t)
                        included_count += 1
                    else:
                        skipped_count += 1
            total_included_count += included_count
            total_skipped_count += skipped_count
            if verbose:
                print(
                    "< Included %d messages and skipped %d"
                    % (included_count, skipped_count)
                )

    if verbose:
        print(
            "Total: Included %d messages and skipped %d"
            % (total_included_count, total_skipped_count)
        )
    return total_included_count, total_skipped_count


def fix_rosbags(bag_folder):
    # Maybe add the message definition
    # https://bitbucket.org/leggedrobotics/darpa_subt/src/master/tools/utils/rosbag_tools/lib/fix_bag_msg_def.py
    active_bags = [str(b) for b in Path(bag_folder).rglob("*.bag.active")]
    print("Start")
    print(len(active_bags))
    for active_bag in active_bags:
        print(active_bag)
        if os.path.exists(active_bag):
            subprocess.run(["rosbag", "reindex", active_bag])
            out_name = active_bag.replace(".bag.active", ".bag")
            subprocess.run(["mv", active_bag, out_name])


def merge_bags_all(bag_folder, clean_folder, verbose=False):
    fix_rosbags(bag_folder)

    bag_names = np.array([str(b) for b in Path(bag_folder).rglob("*.bag")])
    bag_keys = np.array(
        [b.split("/")[-1][: int(b.split("/")[-1].rfind("_"))] for b in bag_names]
    )

    root_dir = str(Path(bag_names[0]).parent)
    total_included_counts, total_skipped_counts = [], []

    for k in np.unique(bag_keys):
        paths = bag_names[k == bag_keys]
        output_bag = os.path.join(root_dir, k + ".bag")

        total_included_count, total_skipped_count = merge_bags_single(
            input_bag=paths.tolist(), output_bag=output_bag, topics="*", verbose=verbose
        )

        total_included_counts.append(total_included_count)
        total_skipped_counts.append(total_skipped_count)

    for j, k in enumerate(np.unique(bag_keys)):
        print(
            f"For bag {k} total_included_count {total_included_counts[j]} total_skipped_count {total_skipped_counts[j]}"
        )

        if clean_folder:
            paths = bag_names[k == bag_keys]
            for p in paths:
                print(f"   -- {p}")
            print(f"Do you want to delete all used bags (yes/no):")
            x = input()
            if x == "yes":
                for p in paths:
                    os.system(f"rm {p}")
