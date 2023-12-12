#!/usr/bin/env python

import rospy

from dynamic_reconfigure.server import Server
from dynamic_params_bev.cfg import ParamsConfig


def callback(config, level):
    rospy.set_param("dynamic_params_bev/LOWER_LIM", config.LOWER_LIM)
    rospy.set_param("dynamic_params_bev/UPPER_LIM", config.UPPER_LIM)
    rospy.set_param("dynamic_params_bev/IDX", config.IDX)
    return config


if __name__ == "__main__":
    rospy.init_node("dynamic_params_bev", anonymous=False)

    srv = Server(ParamsConfig, callback)
    rospy.spin()
