from typing import Any
from abc import ABC, abstractmethod
import numpy as np


def get_tf_and_header_dict(tf_listener, header: Any, ref_frame: str, tar_frame: str):
    try:
        trans, rot = tf_listener.lookupTransform(ref_frame, tar_frame, header.stamp)
    except:
        print(f"Failed LookUp, {ref_frame}, {tar_frame}")
        return {}, False

    res = {}
    res["header_seq"] = header.seq
    res["header_stamp_nsecs"] = header.stamp.nsecs
    res["header_stamp_secs"] = header.stamp.secs
    res["header_frame_id"] = header.frame_id
    res["tf_translation"] = np.array(trans)
    res["tf_rotation_xyzw"] = np.array(rot)

    return res, True


class Converter(ABC):
    @abstractmethod
    def write_to_h5(
        self, msg: Any, dataset_writer, dataset_key: str, tf_listener
    ) -> bool:
        raise NotImplementedError

    @property
    def msg_type(self) -> Any:
        return Any
