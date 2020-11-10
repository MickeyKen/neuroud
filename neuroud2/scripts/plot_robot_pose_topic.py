#!/usr/bin/env python

import rospy

import sys

import time
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64, Int32, Float64MultiArray

ROBOT_NAME = "ubiquitous_display"
class Subscribe():
    def __init__(self):
        self.x1 = 0.
        self.y1 = 0.
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_xlim(-500,500)
        ax.set_ylim(-500,500)
        ax.set_xticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500])
        ax.set_yticks([-400, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500])
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        rate = rospy.Rate(20)

        # Declaration Subscriber
        # self.view_sub = rospy.Subscriber('/view', Float64MultiArray, self.callback)
        self.sub_odom = rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)

        while not rospy.is_shutdown():
            line1, = ax.plot(int(self.x1*100),int(self.y1*100), '*', color="red")

            plt.pause(0.01)
            line1.remove()

            rate.sleep()

    def callback(self, pose):
        self.x1 = pose.pose[pose.name.index(ROBOT_NAME)].position.x
        self.y1 = pose.pose[pose.name.index(ROBOT_NAME)].position.y



if __name__ == '__main__':
    rospy.init_node('view_robot_pose')

    Subscribe = Subscribe()

    rospy.spin()
