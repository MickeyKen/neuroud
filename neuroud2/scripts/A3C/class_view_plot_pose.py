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

class Subscribe():
    def __init__(self):
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.x3 = 0.
        self.y3 = 0.
        self.x4 = 0.
        self.y4 = 0.
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_xlim(-400,400)
        ax.set_ylim(-400,400)
        ax.set_xticks([-400, -300, -200, -100, 0, 100, 200, 300, 400])
        ax.set_yticks([-400, -300, -200, -100, 0, 100, 200, 300, 400])
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        rate = rospy.Rate(20)

        # Declaration Subscriber
        self.view_sub = rospy.Subscriber('/view', Float64MultiArray, self.callback)

        while not rospy.is_shutdown():
            line1, = ax.plot(int(self.y1*100),-int(self.x1*100), 'o', color="red")
            line2, = ax.plot(int(self.y2*100),-int(self.x2*100), '*', color="red")
            line3, = ax.plot(int(self.y3*100),-int(self.x3*100), 'o', color="blue")
            line4, = ax.plot(int(self.y4*100),-int(self.x4*100), '*', color="blue")

            plt.pause(0.01)
            line1.remove()
            line2.remove()
            line3.remove()
            line4.remove()

            rate.sleep()

    def callback(self, data):
        self.x1 = data.data[0]
        self.y1 = data.data[1]
        self.x2 = data.data[2]
        self.y2 = data.data[3]
        self.x3 = data.data[4]
        self.y3 = data.data[5]
        self.x4 = data.data[6]
        self.y4 = data.data[7]


if __name__ == '__main__':
    rospy.init_node('detect_optimize_point_server')

    Subscribe = Subscribe()

    # rospy.Subscriber('/people_tracker_measurements', PositionMeasurementArray , server.ptm_callback)
    # rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, server.pose_callback)

    rospy.spin()
