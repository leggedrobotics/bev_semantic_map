import numpy as np
import argparse
import os
from tqdm import tqdm
from time import time as get_time
from rosbag.bag import Bag
from tf_bag import BagTfTransformer
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sortedcontainers import SortedDict
from pathlib import Path
from os.path import join
import subprocess
from utils.h5py_writer import DatasetWriter
from utils.rosbag_merging import merge_bags_single
from collections import deque
from typing import Dict
from converters.base import get_tf_and_header_dict
import copy
import pickle
from utils.ignore_tf_warnings import suppress_TF_REPEATED_DATA
import concurrent.futures

# suppress_TF_REPEATED_DATA()


class BufferElement:
    """
    Class for the Buffer Element. Consists of the msg, timestamp (ts) and h5_idx. Note that the ts
    might differ from the msg header ts and we will always use the msg header ts
    """

    def __init__(self, msg, ts, h5_idx=None):
        self.msg = msg
        self.ts = ts
        self.h5_idx = h5_idx


def set_defaults(config):
    """
    Sets the default values for the converters if they are missing
    """
    default_converter = config.dataset.default_converter
    # Iterate over converters and set defaults for missing keys
    for converter in config.dataset.converters.values():
        for key, value in default_converter.items():
            converter[key] = converter.get(key, value)


def is_buffer_full(buffer_dict: Dict[str, deque]):
    """
    Check if all deques in the given dictionary are full.
    Args:
        buffer_dict (Dict[str, deque]): A dictionary containing deque objects.
    Returns:
        bool: True if all deques are full, False otherwise.
    """
    return all(len(dq) == dq.maxlen for dq in buffer_dict.values())


def find_closest_element(buffer: deque, query_ts):
    """
    Find the closest element in the provided deque of buffer elements w.r.t the query timestamp
    Args:
        buffer (deque[BufferElement]): A deque containing the BufferElements class instances
        query_ts: reference query timestamp
    Returns:
        c_idx (idx of the closest element in buffer), c_diff (difference in the timestamps in seconds)
    """
    c_idx = 0
    try:
        c_diff = abs(buffer[c_idx].msg.header.stamp.to_sec() - query_ts)
    except:
        c_diff = abs(buffer[c_idx].msg.info.header.stamp.to_sec() - query_ts)
    for idx, ele in enumerate(buffer):
        try:
            diff = abs(ele.msg.header.stamp.to_sec() - query_ts)
        except:
            diff = abs(ele.msg.info.header.stamp.to_sec() - query_ts)

        if diff < c_diff:
            c_idx = idx
            c_diff = diff

    return c_idx, c_diff


class BagToH5:
    def __init__(self, directory: str, cfg: str) -> None:
        self.cfg = OmegaConf.load(cfg)
        OmegaConf.resolve(self.cfg)
        set_defaults(self.cfg)
        self.cfg = instantiate(self.cfg)
        self.sequence = str(directory).split("/")[-1]
        print(f"Processing Sequence {self.sequence}")
        # Find the TF bag file
        tf_rosbags = [
            str(s)
            for s in Path(directory).rglob("*.bag")
            if str(s).find(self.cfg.general.tf_keyword) != -1
        ]
        print(tf_rosbags)

        output_bag_tf = join(directory, "merged_tf.bag")

        if len(tf_rosbags) > 1:
            # merge tf bags
            if not os.path.exists(output_bag_tf):
                for p in tf_rosbags:
                    subprocess.run(["rosbag", "reindex", p])
                    subprocess.run(["rosbag", "decompress", p])

                total_included_count, total_skipped_count = merge_bags_single(
                    input_bag=tf_rosbags,
                    output_bag=output_bag_tf,
                    topics="/tf /tf_static",
                    verbose=True,
                )
        else:
            output_bag_tf = tf_rosbags[0]

        print("Start Loading Rosbag Odometry")
        self.tf_listener = BagTfTransformer(output_bag_tf)

        self.bag_keywords = self.cfg.general.bag_keywords
        self.rosbags = {keyword: [] for keyword in self.bag_keywords}

        for root, dirs, files in os.walk(directory):
            for f in files:
                for bag_keyword in self.bag_keywords:
                    if bag_keyword in f:
                        self.rosbags[bag_keyword].append(join(root, f))

        print(self.rosbags)
        # Sort the list of bag files
        for key in self.rosbags.keys():
            self.rosbags[key] = sorted(self.rosbags[key])

        # Dictionary for topics and h5_key as (k,v)
        self.topic_to_h5_keys = {
            x.topic: x.obs_name for x in self.cfg.dataset.converters.values()
        }

        # Dictionary for topics to converter key as (k, v)
        self.topic_to_converter_keys = {
            x.topic: key for key, x in self.cfg.dataset.converters.items()
        }

        for keyword, bag_list in self.rosbags.items():
            print(f"Found {keyword} Rosbags: {bag_list}")

        # Sets for all relevant topics and h5 keys
        self.topics = set([x.topic for x in self.cfg.dataset.converters.values()])
        self.h5_keys = set([x.obs_name for x in self.cfg.dataset.converters.values()])
        self.h5_to_converter_keys = {
            x.obs_name: key for key, x in self.cfg.dataset.converters.items()
        }

        print("H5_to_converter ", self.h5_to_converter_keys)

        os.makedirs(self.cfg.general.dest, exist_ok=True)
        # Instantiate the Dataset writer class
        self.h5py_file = join(self.cfg.general.dest, f"{self.sequence}.h5py")
        self.pkl_cfg_file = join(self.cfg.general.dest, f"{self.sequence}.pkl")
        self.dataset_writer = DatasetWriter(self.h5py_file)
        self.anchor_h5_key = self.cfg.dataset.anchor_obs_name
        self.converters = self.cfg.dataset.converters
        self.anchor_converter = self.cfg.dataset.converters[
            self.h5_to_converter_keys[self.anchor_h5_key]
        ]

        self.ref_frame = self.cfg.dataset.reference_frame
        self.curr_anchor_t = np.zeros(3)
        self.prev_anchor_t = np.zeros(3)
        self.anchor_dist_threshold = self.cfg.dataset.anchor_dist_threshold

        self.max_samples = self.cfg.general.max_samples
        self.early_exit = False
        print(self.topics)

    def cleanup(self, sequence_data, bags):
        for i in range(0, len(sequence_data)):
            if i < len(sequence_data) * 0.7:
                sequence_data[i]["mode"] = "train"
            else:
                sequence_data[i]["mode"] = "val"

        print(f"Saving config to {self.pkl_cfg_file}")
        with open(self.pkl_cfg_file, "wb") as handle:
            pickle.dump(sequence_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for bag_list in bags.values():
            for b in bag_list:
                b.close()

        self.dataset_writer.close()
        self.early_exit = True

    def run(self):
        # Open the rosbags
        t = get_time()
        print("Opening rosbags...")

        # Create dictionary of Bag files by keywords
        bags = {
            bag_key: [Bag(x, "r") for x in bag_list]
            for bag_key, bag_list in self.rosbags.items()
        }

        print(f"Opened rosbags in {get_time() - t:.0f}s")

        # Create the dictionary of buffers
        buffer_dict = {
            x.obs_name: deque(maxlen=x.buffer_size)
            for x in self.converters.values()
            if (not x.store_once and str(x.obs_name) != self.anchor_h5_key)
        }

        # Create dictionary for all store_once topics (Like camera info)
        stored_once_dict = {
            x.obs_name: False for x in self.converters.values() if x.store_once
        }

        # Create set for all store_always topics (Like GridMap)
        store_always_dict = set(
            x.obs_name for x in self.converters.values() if x.store_always
        )

        print("Buffer Dict: ", buffer_dict)

        # Create Buffer for anchor message
        anchor_buffer = deque(maxlen=self.anchor_converter.buffer_size)

        nr_of_msgs = sum(
            x.get_message_count(topic_filters=list(self.topics))
            for bag_list in bags.values()
            for x in bag_list
        )

        print(f"Nr of msgs: {nr_of_msgs}")

        curr_h5_inds = {h5_key: 0 for h5_key in self.h5_keys}
        bag_its = {
            bag_key: [x.read_messages(topics=list(self.topics)) for x in bag_list]
            for bag_key, bag_list in bags.items()
        }

        bag_msgs = SortedDict()

        curr_bag_its = []
        curr_bag_keys = []
        for bag_key, bag_list in bag_its.items():
            curr_bag_its.append(bag_list.pop(0))
            curr_bag_keys.append(bag_key)

        for i, it in enumerate(curr_bag_its):
            t, topic, msg = get_next_msg(it)
            bag_msgs[(t, i)] = (topic, msg)

        print("Got initial Messages")

        # For the pickle config file
        sequence_data = []

        with tqdm(
            total=nr_of_msgs, desc="Converting rosbag to hdf5", unit="msg"
        ) as pbar:
            time = 0.0
            suc_samples = 0
            total_anchor = 0
            while True:
                try:
                    # get next msg
                    (t, bag_idx), (topic, msg) = bag_msgs.popitem(0)
                    if msg is None:
                        break
                    if t < time:
                        raise RuntimeError(
                            f"t ({t}) < time ({time}) ==> msgs not in order (topic: {topic})"
                        )
                    time = t

                    # Take care of the store once message if not already stored
                    if str(self.topic_to_h5_keys[topic]) in stored_once_dict:
                        # Check if the data has already been stored
                        if not stored_once_dict[self.topic_to_h5_keys[topic]]:
                            print(f"Storing data for topic {topic}")
                            # Write the data to H5
                            if self.converters[
                                self.topic_to_converter_keys[topic]
                            ].converter.write_to_h5(
                                msg,
                                self.dataset_writer,
                                self.sequence,
                                self.topic_to_h5_keys[topic],
                                self.tf_listener,
                            ):
                                stored_once_dict[self.topic_to_h5_keys[topic]] = True

                    if self.topic_to_h5_keys[topic] == self.anchor_h5_key:
                        anchor_buffer.append(BufferElement(msg, t, None))
                        total_anchor += 1

                        if is_buffer_full(buffer_dict):
                            curr_anchor_ele = anchor_buffer.popleft()

                            print(f"Processing message {topic} : {curr_anchor_ele.ts}")

                            tar_frame = curr_anchor_ele.msg.header.frame_id
                            tf_dict_ref_header, suc = get_tf_and_header_dict(
                                self.tf_listener,
                                curr_anchor_ele.msg.header,
                                ref_frame=self.ref_frame,
                                tar_frame=tar_frame,
                            )
                            if not suc:
                                is_valid = False
                                continue

                            self.curr_anchor_t = tf_dict_ref_header["tf_translation"]

                            is_valid = True
                            if (
                                np.linalg.norm(self.curr_anchor_t - self.prev_anchor_t)
                                < self.anchor_dist_threshold
                            ):
                                # Doesn't satisfy the anchor dist threshold
                                is_valid = False
                                print("Doesnt satisfy Dist Threshold")
                                continue

                            # Create dictionary of Buffer elements for the current sample
                            curr_sample_dict = {}
                            curr_sample_dict[self.anchor_h5_key] = curr_anchor_ele

                            # Use this dictionary for additional samples. For e.g. 10 pointclouds
                            curr_sample_aux_dict = {}

                            # Query the closest msg
                            for obs_name in buffer_dict.keys():
                                c_idx, c_diff = find_closest_element(
                                    buffer_dict[obs_name],
                                    curr_anchor_ele.msg.header.stamp.to_sec(),
                                )
                                # Check if the difference satisfies the Time threshold
                                if (
                                    c_diff
                                    < self.converters[
                                        self.h5_to_converter_keys[obs_name]
                                    ].max_ts
                                ):
                                    print(
                                        f"{obs_name} satisfies Timestamp threshold {c_diff} with idx {c_idx}"
                                    )
                                    curr_sample_dict[obs_name] = buffer_dict[obs_name][
                                        c_idx
                                    ]

                                    # Check if it requires extra samples
                                    merge_N = self.converters[
                                        self.h5_to_converter_keys[obs_name]
                                    ].merge_N_temporal
                                    if merge_N != 0:
                                        try:
                                            # Aggregate either from the past (-ve) or the future (+ve)
                                            if len(list(merge_N)) == 1:
                                                val = 1 * np.sign(merge_N)
                                                curr_sample_aux_dict[obs_name] = list(
                                                    buffer_dict[obs_name]
                                                )[c_idx + val : c_idx + merge_N : val]
                                                print(
                                                    f"Temporally merging {merge_N - 1} messages from buffer for the obs {obs_name}"
                                                )
                                            # Aggregate from both past and future
                                            elif len(list(merge_N)) == 2:
                                                merge_N = list(merge_N)
                                                temp_list = list(buffer_dict[obs_name])
                                                if merge_N[0] < 0 and merge_N[1] > 0:
                                                    curr_sample_aux_dict[
                                                        obs_name
                                                    ] = temp_list[
                                                        c_idx + merge_N[0] : c_idx
                                                    ]
                                                    curr_sample_aux_dict[
                                                        obs_name
                                                    ] += temp_list[
                                                        c_idx + 1 : c_idx + merge_N[1]
                                                    ]
                                                else:
                                                    curr_sample_aux_dict[
                                                        obs_name
                                                    ] = list(buffer_dict[obs_name])[
                                                        c_idx
                                                        + merge_N[0] : c_idx
                                                        + merge_N[1]
                                                    ]

                                                # print(list(buffer_dict[obs_name][:10]))

                                                print(
                                                    f"Temporally merging {merge_N[1] - merge_N[0] - 1} messages from buffer for the obs {obs_name}"
                                                )

                                            # curr_sample_aux_dict[obs_name] = list(
                                            #     buffer_dict[obs_name]
                                            # )[c_idx - 1 : c_idx - merge_N : -1]
                                            # print(
                                            #     f"Temporally merging {merge_N-1} messages from buffer for the obs {obs_name}"
                                            # )

                                        except:
                                            print(
                                                f"c_idx is {c_idx} and buffer length is {len(buffer_dict[obs_name])}"
                                            )
                                            print(
                                                f"Unable to query {merge_N} messages from buffer for the obs {obs_name}"
                                            )

                                else:
                                    is_valid = False
                                    print(
                                        f"{obs_name} doesn't satisfy Timestamp threshold {c_diff} with idx {c_idx}"
                                    )
                                    break

                            if not is_valid:
                                continue
                            # All topics satisfy the time threshold
                            # Iterate through the curr sample dictionary
                            # If h5_idx is None -> call the target functions to write to H5 file
                            # Else -> (Data is already written in H5 file) Just take care of the indices for the config file

                            # Take care of the extra samples which need to be written

                            # Create the dict for the pickle config file
                            sample = {}
                            sample_valid = True
                            for key, val in curr_sample_dict.items():
                                if val.h5_idx is None:
                                    # print(f"value h5 idx is None for {key}")
                                    #  Call the target functions to write to H5 file
                                    success = self.converters[
                                        self.h5_to_converter_keys[key]
                                    ].converter.write_to_h5(
                                        val.msg,
                                        self.dataset_writer,
                                        self.sequence,
                                        key,
                                        self.tf_listener,
                                    )
                                    # print(f"success is {success}")
                                    if success:
                                        # Handle the H5 idx
                                        val.h5_idx = curr_h5_inds[key]
                                        # If auxilliary, then add as list
                                        if (
                                            self.converters[
                                                self.h5_to_converter_keys[key]
                                            ].merge_N_temporal
                                        ) != 0:
                                            sample[key] = [curr_h5_inds[key]]
                                        else:
                                            sample[key] = curr_h5_inds[key]

                                        curr_h5_inds[key] += 1

                                    else:
                                        sample_valid = False
                                        break

                                else:
                                    if (
                                        self.converters[
                                            self.h5_to_converter_keys[key]
                                        ].merge_N_temporal
                                    ) != 0:
                                        sample[key] = [val.h5_idx]
                                    else:
                                        sample[key] = val.h5_idx

                            # Handle the Auxilliary writing
                            if sample_valid:
                                for key, val in curr_sample_aux_dict.items():
                                    if sample_valid:
                                        for aux_item in val:
                                            if aux_item.h5_idx is None:
                                                success = self.converters[
                                                    self.h5_to_converter_keys[key]
                                                ].converter.write_to_h5(
                                                    aux_item.msg,
                                                    self.dataset_writer,
                                                    self.sequence,
                                                    key,
                                                    self.tf_listener,
                                                )

                                                if success:
                                                    # Handle the H5 idx
                                                    aux_item.h5_idx = curr_h5_inds[key]
                                                    sample[key].append(
                                                        curr_h5_inds[key]
                                                    )
                                                    curr_h5_inds[key] += 1
                                                else:
                                                    sample_valid = False
                                                    break

                                            else:
                                                sample[key].append(aux_item.h5_idx)

                                    # print(f"Aux {key} has the indices {sample[key]}")
                                    sample[key] = sample[key][::-1]

                            if sample_valid:
                                suc_samples += 1
                                print(f"Samples are {suc_samples}")
                                self.prev_anchor_t = self.curr_anchor_t
                                sample["sequence_key"] = self.sequence
                                sequence_data.append(copy.deepcopy(sample))

                                if (
                                    self.max_samples != "None"
                                    and suc_samples > int(self.max_samples) - 1
                                ):
                                    # Cleanup, store pkl config file and return from function
                                    print(
                                        f"No. of successful samples are {suc_samples} / {total_anchor}"
                                    )
                                    print(
                                        f"Reached maximum number of samples for sequence {self.sequence}"
                                    )
                                    self.cleanup(sequence_data, bags)
                                    return

                    elif (
                        self.topic_to_h5_keys[topic] in buffer_dict.keys()
                    ):  # The Data is not Store Once
                        h5_key = self.topic_to_h5_keys[topic]
                        # print(f"Adding to buffer {topic} : {t}")
                        buffer_dict[h5_key].append(BufferElement(msg, t, None))
                        # Check if the type is always store
                        if h5_key in store_always_dict:
                            suc = self.converters[
                                self.h5_to_converter_keys[h5_key]
                            ].converter.write_to_h5(
                                msg,
                                self.dataset_writer,
                                self.sequence,
                                h5_key,
                                self.tf_listener,
                            )
                            if suc:
                                buffer_dict[h5_key][-1].h5_idx = curr_h5_inds[h5_key]
                                curr_h5_inds[h5_key] += 1

                finally:
                    if not self.early_exit:
                        t, topic, msg = get_next_msg(curr_bag_its[bag_idx])
                        if msg is None:
                            # Need to update the current bag_its
                            try:
                                curr_bag_its[bag_idx] = bag_its[
                                    curr_bag_keys[bag_idx]
                                ].pop(0)
                                # Now again try to extract the next msg
                                t, topic, msg = get_next_msg(curr_bag_its[bag_idx])
                            except:
                                print(
                                    f"Finished bag files for {curr_bag_keys[bag_idx]}"
                                )

                        bag_msgs[(t, bag_idx)] = (topic, msg)
                        pbar.update(1)

        print(f"No. of successful samples are {suc_samples} / {total_anchor}")
        self.cleanup(sequence_data, bags)


def get_next_msg(bag_it):
    try:
        topic, msg, t = next(bag_it)
        return t.to_sec(), topic, msg
    except StopIteration:
        return float("inf"), "", None


def process_folder(folder, cfg):
    rosbag_converter = BagToH5(folder, cfg)
    rosbag_converter.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rosbags to HDF5")
    parser.add_argument("--cfg", type=str, help="path to config file")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    if not cfg.general.process_multiple:
        # Init the BagtoH5 class
        rosbag_converter = BagToH5(cfg.general.rosbags, args.cfg)
        rosbag_converter.run()

    else:
        # List all directories
        dir_list = [
            os.path.join(cfg.general.rosbags, folder)
            for folder in os.listdir(cfg.general.rosbags)
        ]
        print(f"Directory list is {dir_list}")

        max_workers = cfg.general.max_threads

        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(process_folder, folder, args.cfg)
                    for folder in dir_list
                ]
                # Wait for all threads to complete and collect any exceptions
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in one of the threads: {e}")

        else:
            for folder in dir_list:
                process_folder(folder, args.cfg)

    print("Finished processing all files")
