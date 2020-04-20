#!/usr/bin/env python

import rospy

from gazebo_msgs.srv import GetModelState, GetModelStateRequest

import sys

import math
import numpy as np
import matplotlib.pyplot as plt

def get_pose(name):
    set = GetModelStateRequest()
    set.model_name = name
    response = call(set)
    return response.pose

if __name__ == '__main__':

    rospy.init_node('view_node')

    call = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    human_number = 5

    name_list = []
    name_list.append("ubiquitous_display")

    for i in range(human_number):
        name_list.append("actor" + str(i))

    plt.ion()
    plt.title('HEATMAP')
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    plt.grid()

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():

        for n in name_list:
            pose = get_pose(n)
            x = pose.position.x*100
            y = pose.position.y*100

            # print (pose)
        plt.plot(int(y),int(x), 'o', color="blue")
        plt.draw()
        plt.pause(0.1)
        rate.sleep()
