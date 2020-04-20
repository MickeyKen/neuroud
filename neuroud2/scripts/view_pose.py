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
    # name_list.append("ubiquitous_display")

    for i in range(human_number):
        name_list.append("actor" + str(i))

    plt.ion()
    plt.title('HEATMAP')

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        plt.xlim(-500,500)
        plt.ylim(-500,500)
        plt.grid()

        pose = get_pose("ubiquitous_display")
        x = pose.position.x*100
        y = pose.position.y*100
        plt.plot(int(y),int(x), '^', color="red")

        for n in name_list:
            pose = get_pose(n)
            x = pose.position.x*100
            y = pose.position.y*100
            plt.plot(int(y),int(x), 'o', color="blue")


            # print (pose)
        plt.draw()
        plt.pause(0.01)
        plt.clf()
        rate.sleep()
