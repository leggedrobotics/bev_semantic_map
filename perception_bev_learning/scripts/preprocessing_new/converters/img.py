import numpy as np
from .base import Converter, get_tf_and_header_dict
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from typing import Any, Optional, Union, Tuple, List
import cv2


class CompressedImgConverter(Converter):
    def __init__(
        self,
        reference_frame=None,
        aux_target_frame: Optional[List[str]] = None,
        aux_tfs: Optional[List[Tuple[str, str]]] = None,
        resize: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.bridge = CvBridge()
        if resize == "None":
            resize = None
        if isinstance(resize, int):
            resize = (resize, resize)
        self.resize = resize
        self.ref_frame = reference_frame
        self.aux_target_frame = aux_target_frame
        self.aux_tfs = aux_tfs

    def msg_type(self) -> Any:
        return CompressedImage

    def write_to_h5(
        self, msg: Any, dataset_writer, seq_name: str, dataset_key: str, tf_listener
    ) -> bool:
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if self.resize is not None:
            img = cv2.resize(img, (self.resize[0], self.resize[1]))

        res_dict = {"image": img}

        tf_exists = True

        if self.ref_frame is not None:
            # Query the TF from header target frame to reference frame
            tar_frame = msg.header.frame_id
            tf_dict_ref_header, suc = get_tf_and_header_dict(
                tf_listener, msg.header, ref_frame=self.ref_frame, tar_frame=tar_frame
            )
            tf_exists = tf_exists and suc
            if not tf_exists:
                return False
            res_dict.update(tf_dict_ref_header)

            if self.aux_target_frame is not None:
                # Query the TFs from all aux target frames to reference frame
                for t_frame in self.aux_target_frame:
                    dict_key_t = f"tf_translation_{str(self.ref_frame).split('/')[-1]}__{str(t_frame).split('/')[-1]}"
                    dict_key_r = f"tf_rotation_xyzw_{str(self.ref_frame).split('/')[-1]}__{str(t_frame).split('/')[-1]}"
                    tf_dict, suc = get_tf_and_header_dict(
                        tf_listener,
                        msg.header,
                        ref_frame=self.ref_frame,
                        tar_frame=t_frame,
                    )
                    tf_exists = tf_exists and suc
                    if not tf_exists:
                        return False
                    res_dict[dict_key_t] = tf_dict["tf_translation"]
                    res_dict[dict_key_r] = tf_dict["tf_rotation_xyzw"]

            if self.aux_tfs is not None:
                for ref, tgt in self.aux_tfs:
                    dict_key_t = f"tf_translation_{str(ref).split('/')[-1]}__{str(tgt).split('/')[-1]}"
                    dict_key_r = f"tf_rotation_xyzw_{str(ref).split('/')[-1]}__{str(tgt).split('/')[-1]}"
                    tf_dict, suc = get_tf_and_header_dict(
                        tf_listener,
                        msg.header,
                        ref_frame=ref,
                        tar_frame=tgt,
                    )
                    tf_exists = tf_exists and suc
                    if not tf_exists:
                        return False
                    res_dict[dict_key_t] = tf_dict["tf_translation"]
                    res_dict[dict_key_r] = tf_dict["tf_rotation_xyzw"]

        if tf_exists:
            fieldname = dataset_key
            static_keys = ["header_frame_id"]
            static_dict = {k: v for k, v in res_dict.items() if k in static_keys}
            dataset_writer.add_static(seq_name, fieldname, static_dict)
            dynamic_dict = {k: v for k, v in res_dict.items() if k not in static_keys}
            dataset_writer.add_data(seq_name, fieldname, dynamic_dict)
            return True
        else:
            return False  # TF doesn;t exist
