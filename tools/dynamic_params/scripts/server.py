#!/usr/bin/env python

import rospy

from dynamic_reconfigure.server import Server
from dynamic_params.cfg import ParamsConfig


def callback(config, level):
    # rospy.loginfo("""Reconfigure Request: {LOWER_LIM} - {UPPER_LIM}""".format(**config))
    rospy.set_param("dynamic_params/LOWER_LIM", config.LOWER_LIM)
    rospy.set_param("dynamic_params/UPPER_LIM", config.UPPER_LIM)
    rospy.set_param("dynamic_params/IDX", config.IDX)
    return config


if __name__ == "__main__":
    rospy.init_node("dynamic_params", anonymous=False)

    srv = Server(ParamsConfig, callback)
    rospy.spin()
