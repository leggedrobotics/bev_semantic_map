import numpy as np
from .base import Converter, get_tf_and_header_dict
from sensor_msgs.msg import PointCloud2
from typing import Any, Optional, Union, Tuple, List
import cv2

PDC_DATATYPE = {
    "1": np.int8,
    "2": np.uint8,
    "3": np.int16,
    "4": np.uint16,
    "5": np.int32,
    "6": np.uint32,
    "7": np.float32,
    "8": np.float64,
}


class PointCloudConverter(Converter):
    def __init__(
        self,
        reference_frame=None,
        aux_target_frame: Optional[List[str]] = None,
        aux_tfs: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__()

        self.ref_frame = reference_frame
        self.aux_target_frame = aux_target_frame
        self.aux_tfs = aux_tfs

    def msg_type(self) -> Any:
        return PointCloud2

    def write_to_h5(
        self, msg: Any, dataset_writer, seq_name: str, dataset_key: str, tf_listener, *args, **kwargs
    ) -> bool:
        
        fieldname = dataset_key

        res_dict = {}
        res_tf_dict = {}

        res = np.frombuffer(msg.data, np.dtype(np.int8)).reshape((msg.width, -1))

        for field in msg.fields:
            if field.name not in ["x", "y", "z", "intensity"]:
                continue
            assert field.count == 1, "If this is not case, maybe does not work"
            PDC_DATATYPE[str(field.datatype)]
            dtype = PDC_DATATYPE[str(field.datatype)]
            nbytes = dtype(1).nbytes
            data = res[:, field.offset : field.offset + nbytes]
            res_dict[field.name] = np.frombuffer(data.copy(), np.dtype(dtype)).reshape(
                (-1)
            )

            res_dict["valid"] = np.ones((res_dict[field.name].shape[0],), dtype=bool)

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
            res_tf_dict.update(tf_dict_ref_header)

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
                    res_tf_dict[dict_key_t] = tf_dict["tf_translation"]
                    res_tf_dict[dict_key_r] = tf_dict["tf_rotation_xyzw"]

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
                    res_tf_dict[dict_key_t] = tf_dict["tf_translation"]
                    res_tf_dict[dict_key_r] = tf_dict["tf_rotation_xyzw"]

        if tf_exists:
            # res_dict.update(tf_dict)
            static_keys = ["header_frame_id"]
            static_dict = {k: v for k, v in res_tf_dict.items() if k in static_keys}
            dataset_writer.add_static(seq_name, fieldname, static_dict)

            dynamic_dict = {k: v for k, v in res_tf_dict.items() if k not in static_keys}
            dataset_writer.add_data(seq_name, fieldname, dynamic_dict)
            dataset_writer.add_pointcloud(seq_name, fieldname, res_dict)

            return True
        else:
            return False  # TF doesn;t exist