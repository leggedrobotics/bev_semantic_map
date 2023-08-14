#!/usr/bin/env python

"""
Prints the tf from a source frame to a target frame.

Author: Robin Schmid
Date: Nov 2022
"""

import tf
import rospy

if __name__ == "__main__":
    print("Start listening to tf")

    rospy.init_node("tf_printer")

    SRC_FRAME = "bpearl_rear"  # depth_camera_rear_depth_optical_frame, lidar
    DST_FRAME = "cam4_sensor_frame_helper"  # depth_camera_rear_depth_optical_frame, cam4_sensor_frame_helper

    listener = tf.TransformListener()

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform(SRC_FRAME, DST_FRAME,
                                                    rospy.Time.now())
            # realsense to cam4
            # trans = [0.013814848502623628, -0.20254032233291452, -0.20367474543340167]
            # rot = [-0.24981959957596894, 0.011532330452940665, 0.007644459970728101, -0.9681935422495768]

            # lidar to cam4
            # trans = [0.0021375334062859854, -0.03956706739849679, 0.0920467808]
            # rot = [0.0087744421888676, -0.7068219147571273, 0.7073014361400815, -0.007104112718762427]

            # perugia lidar to cam4
            # trans = [0.0036920364377833652, -0.04315291620837264, 0.08874986509999999]
            # rot = [0.009399369655087067, -0.7057808978358807, 0.7083022231566545, -0.009640371953725439]

            # perugia pbearl_rear to cam4
            # trans = [-0.16591687972787286, -0.160695551724784, -0.022342083791627387]
            # rot = [0.0035049818099763264, 0.013120576338846446, 0.382490692842936, -0.923859532324898]

            print((trans, rot))
        except:
            pass
