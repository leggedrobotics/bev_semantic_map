#!/usr/bin/env python

import rospy

from dynamic_reconfigure.server import Server
from dynamic_params.cfg import ParamsConfig


def callback(config, level):
    rospy.loginfo("""Reconfigure Request: {int_param}""".format(**config))
    return config


if __name__ == "__main__":
    rospy.init_node("dynamic_params", anonymous=False)

    srv = Server(ParamsConfig, callback)
    rospy.spin()
