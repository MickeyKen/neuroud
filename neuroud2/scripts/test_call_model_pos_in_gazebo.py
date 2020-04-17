#!/usr/bin/env python

import rospy

from gazebo_msgs.srv import GetModelState, GetModelStateRequest

import sys


if __name__ == '__main__':

    rospy.init_node('call_model_pos_node')

    call = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    set = GetModelStateRequest()

    set.model_name = "actor"

    response = call(set)

    if response.success:
        print(response)
    else:
        rospy.logerr('failed')
