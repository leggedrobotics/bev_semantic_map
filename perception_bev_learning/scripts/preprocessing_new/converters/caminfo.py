import numpy as np
from .base import Converter
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from typing import Any, Optional, Union, Tuple, List
import cv2


class CameraInfoConverter(Converter):
    def __init__(
        self,
        reference_frame=None,
        aux_target_frame: Optional[List[str]] = None,
        resize: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        super().__init__()
        if resize == "None":
            resize = None
        if isinstance(resize, int):
            resize = (resize, resize)
        self.resize = resize

    def msg_type(self) -> Any:
        return CameraInfo

    def write_to_h5(
        self, msg: Any, dataset_writer, seq_name: str, dataset_key: str, tf_listener
    ) -> bool:
        fieldname = dataset_key
        camera_info = {
            method_name: getattr(msg, method_name)
            for method_name in dir(type(msg))
            if not callable(getattr(type(msg), method_name))
            and method_name[0] != "_"
            and method_name.find("roi") == -1
            and method_name.find("header") == -1
        }
        for k, v in camera_info.items():
            if type(v) is tuple:
                camera_info[k] = np.array(list(v))

        if self.resize is not None:
            factor = camera_info["width"] / self.resize[0]

            camera_info["K"][:6] = np.array(camera_info["K"][:6]) / factor
            camera_info["P"][:8] = np.array(camera_info["P"][:8]) / factor
            camera_info["P"][-1] = np.array(camera_info["P"][-1]) / factor
            camera_info["height"] = int(camera_info["height"] / factor)
            camera_info["width"] = int(camera_info["width"] / factor)

        try:
            dataset_writer.add_static(seq_name, fieldname, camera_info)
            return True

        except:
            return False
