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
        self, msg: Any, dataset_writer, seq_name: str, dataset_key: str, tf_listener, *args, **kwargs
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
            sx = self.resize[0] / camera_info["width"]
            sy = self.resize[1] / camera_info["height"]

            K_orig = np.array(camera_info["K"]).reshape((3,3))
            P_orig = np.array(camera_info["P"]).reshape((3,4))

            RT_orig = np.linalg.inv(K_orig) @ P_orig

            # K_orig[0,0] = 512.11257225
            # K_orig[1,1] = 502.77627934
            # K_orig[0,2] = 968.75886685
            # K_orig[1,2] = 644.72345463
            
            K_new = K_orig.copy()
            K_new[0, 0] *= sx
            K_new[1, 1] *= sy
            K_new[0, 2] *= sx
            K_new[1, 2] *= sy

            camera_info["K"] = K_new.reshape(-1)

            P_new = K_new @ RT_orig

            camera_info["P"] = P_new.reshape(-1)

            camera_info["height"] = int(camera_info["height"] * sy)
            camera_info["width"] = int(camera_info["width"] * sx)

        try:
            dataset_writer.add_static(seq_name, fieldname, camera_info)
            return True

        except:
            return False
