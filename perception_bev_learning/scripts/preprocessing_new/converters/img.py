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
        info_obs_name: Optional[str] = None,
        undistort: Optional[bool] = False,
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
        self.info_obs_name = info_obs_name
        self.undistort = undistort
        self.D = None
        self.K = None
        self.distortion_model = None

    def msg_type(self) -> Any:
        return CompressedImage

    def write_to_h5(
        self, msg: Any, dataset_writer, seq_name: str, dataset_key: str, tf_listener, info_dict, *args, **kwargs
    ) -> bool:
        img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

        if self.undistort:
            if self.info_obs_name is not None:
                if self.D is None:
                    # Extract the Distortion Matrix 
                    msg_cam_info = info_dict[self.info_obs_name]
                    camera_info = {
                        method_name: getattr(msg_cam_info, method_name)
                        for method_name in dir(type(msg_cam_info))
                        if not callable(getattr(type(msg_cam_info), method_name))
                        and method_name[0] != "_"
                        and method_name.find("roi") == -1
                        and method_name.find("header") == -1
                    }
                    for k, v in camera_info.items():
                        if type(v) is tuple:
                            camera_info[k] = np.array(list(v))

                    # print(camera_info["D"])
                    # print(camera_info["distortion_model"])
                    self.D = camera_info["D"]
                    self.K = camera_info["K"].reshape((3,3))
                    # print(img.shape)
                    self.h = img.shape[0]
                    self.w = img.shape[1]
                    # Need to update to original K
                    print(f"Original K is {self.K}")
                    # self.K[0,0] = 512.11257225
                    # self.K[1,1] = 502.77627934
                    # self.K[0,2] = 968.75886685
                    # self.K[1,2] = 644.72345463

                    # self.D = np.array([0.13058332, 0.01104646, 0.01195079, -0.00302817])

                    self.distortion_model = camera_info["distortion_model"]
                    self.K_new, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (self.w, self.h), 1, (self.w, self.h))
                    # print(f"Height is {self.h}, width is {self.w} ")
                    # self.K_new = self.K.copy()  # Optionally, you can define a new camera matrix

                # Perform the undistortion
                print(f"K mat is {self.K}")
                print(f"D mat is {self.D}")
                # img = cv2.undistort(img, self.K, self.D, None, self.K_new)

                # mapx, mapy = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K_new, (self.w, self.h), 5)
                # img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K_new, (self.w, self.h), cv2.CV_16SC2)
                img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

            else:
                print(f"Warning: Cam info matrix does not exist")
                return False

        if self.resize is not None:
            # undistorted_img = cv2.resize(undistorted_img, (self.resize[0], self.resize[1]))
            img = cv2.resize(img, (self.resize[0], self.resize[1]))
        
        # cv2.imshow("Undistorted Image", undistorted_img)
        # cv2.imshow("Original Image", img)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()

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
