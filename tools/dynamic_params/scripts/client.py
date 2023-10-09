#!/usr/bin/env python
import rosparam
import rospy

import dynamic_reconfigure.client


def callback(config):
    rospy.loginfo("Config set to {LOWER_LIM} - {UPPER_LIM}".format(**config))


if __name__ == "__main__":
    rospy.init_node("dynamic_client")

    r = rospy.Rate(1)  # Time to reset dynamic reconfigure
    while not rospy.is_shutdown():
        a = rospy.get_param("dynamic_params/LOWER_LIM")
        b = rospy.get_param("dynamic_params/UPPER_LIM")
        r.sleep()