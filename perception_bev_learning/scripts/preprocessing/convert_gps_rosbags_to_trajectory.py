import os
from pathlib import Path
import argparse
import rospy
import numpy as np
from tqdm import tqdm
import rosbag
from os.path import join
import subprocess
import warnings
import gpxpy
from perception_bev_learning.preprocessing import BagTfTransformerWrapper, get_bag_info
# from perception_bev_learning.utils import load_env, load_pkl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=str,
        default="/media/jonfrey/Data2/bev_traversability/2022-06-07-jpl6_camp_roberts_d2",
        help="Gps bag",
    )
    args = parser.parse_args()
    valid_topics = "/rtk_gps_driver/position_receiver_0/ros/navsatfix"

    from pathlib import Path

    bag_paths = [str(s) for s in Path(args.directory).rglob("*npc*")]

    # env = load_env()
    # dataset_config = load_pkl(os.path.join(env["dataset_root_dir"], "dataset_config_subsample_20.pkl"))
    # for mode in ["train", "val", "test"]:
    #     print(mode)
    #     for l in np.unique([d["sequence_key"] for d in dataset_config if d["mode"] == mode]).tolist():
    #         print(f"  {l}")
    print(bag_paths)

    for bag_path in bag_paths:
        gpx = gpxpy.gpx.GPX()
        # Create first track in our GPX:
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)

        # Create first segment in our GPX track:
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)
        # Create points

        with rosbag.Bag(bag_path, "r") as bag:
            sequence = bag_path.split("/")[-2]
            try:
                start_time = rospy.Time.from_sec(bag.get_start_time())
            except:
                print("Skiped the bag given that it is empty")
                exit - 1

            end_time = rospy.Time.from_sec(bag.get_end_time())

            rosbag_info_dict = get_bag_info(bag_path)
            total_msgs = sum(
                [
                    x["messages"]
                    for x in rosbag_info_dict["topics"]
                    if x["topic"] in valid_topics
                ]
            )
            with tqdm(
                total=total_msgs,
                desc="Total",
                colour="green",
                position=1,
                bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
            ) as pbar:
                for topic, msg, ts in bag.read_messages(
                    topics=valid_topics, start_time=start_time, end_time=end_time
                ):
                    pbar.update(1)
                    if msg.latitude != 0:
                        # Add points to the GPS path
                        gpx_segment.points.append(
                            gpxpy.gpx.GPXTrackPoint(
                                latitude=msg.latitude,
                                longitude=msg.longitude,
                                elevation=msg.altitude,
                            )
                        )
            # print("done")
        print("Created GPX:", gpx.to_xml())

        # modes = [d["mode"] for d in dataset_config if d["sequence_key"].find(sequence) != -1]
        # if len(modes) == 0:
        #     mode = "undefined"
        # else:
        #     mode = modes[0]

        # with open(f"assets/gps/{mode}/{sequence}.gpx", "w") as f:
        #     f.write(gpx.to_xml())

        with open(
            f"/home/patelm/Data/nature_hiking/ARCHE-experiments/long_navigation_gpt/{bag_path.split('/')[-1]}.gpx", "w"
        ) as f:
            f.write(gpx.to_xml())
