import yaml
import subprocess
from tf_bag import BagTfTransformer


def get_bag_info(rosbag_path: str) -> dict:
    # This queries rosbag info using subprocess and get the YAML output to parse the topics
    info_dict = yaml.safe_load(
        subprocess.Popen(["rosbag", "info", "--yaml", rosbag_path], stdout=subprocess.PIPE).communicate()[0]
    )
    return info_dict


class BagTfTransformerWrapper:
    def __init__(self, bag):
        self.tf_listener = BagTfTransformer(bag)

    def waitForTransform(self, parent_frame, child_frame, time, duration):
        return self.tf_listener.waitForTransform(parent_frame, child_frame, time)

    def lookupTransform(self, parent_frame, child_frame, time):
        try:
            return self.tf_listener.lookupTransform(parent_frame, child_frame, time)
        except:
            return (None, None)
