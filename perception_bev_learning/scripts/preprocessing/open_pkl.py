import argparse
from perception_bev_learning.utils import load_pkl
from perception_bev_learning.dataset.bev_dataset import get_sequence_key

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--pkl_file",
        type=str,
        help="pickle file to be read",
        default="/data/bev_traversability/trial_1/racer-jpl9_hw_2023-08-29-21-48-16-UTC_halter-ranch-dubost-2_ROBOT_dubost-datacollect-patrick-1_DS/dataset_config.pkl",
    )

    args = vars(parser.parse_args())

    if args["pkl_file"] == "nan":
        print("Please provide the path to the pickle file to be read")
        exit()
    else:
        dataset_config = load_pkl(args["pkl_file"])

    dataset_config = [d for d in dataset_config if d["mode"] == "train"]
    dataset_config = [
        d for d in dataset_config if d["sequence_key"].find("18-21-54") == -1
    ]

    print(len(dataset_config))

    datum = dataset_config[0]
    sk = datum["sequence_key"]
    sk_new = get_sequence_key(sk)

    print("hello")
