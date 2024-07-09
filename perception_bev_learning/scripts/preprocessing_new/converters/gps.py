import numpy as np
from .base import Converter
from typing import Any, Optional, Union, Tuple, List


class GPSPosConverter(Converter):
    def __init__(
        self,
    ):
        super().__init__()

    def msg_type(self) -> Any:
        # TODO Fix this
        return "GPS Data"

    def write_to_h5(
        self, msg: Any, dataset_writer, seq_name: str, dataset_key: str, tf_listener
    ) -> bool:
        fieldname = dataset_key
        res_dict = {
            "latitude": msg.latitude,
            "longitude": msg.longitude,
            "altitude": msg.altitude,
        }
        dataset_writer.add_data(seq_name, fieldname, res_dict)
        return True


class GPSOrientationConverter(Converter):
    def __init__(
        self,
    ):
        super().__init__()

    def msg_type(self) -> Any:
        # TODO Fix this
        return "GPS Data"

    def write_to_h5(
        self, msg: Any, dataset_writer, seq_name: str, dataset_key: str, tf_listener
    ) -> bool:
        fieldname = dataset_key
        rot = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]
        res_dict = {"rotation_xyzw": np.array(rot)}
        dataset_writer.add_data(seq_name, fieldname, res_dict)
        return True
