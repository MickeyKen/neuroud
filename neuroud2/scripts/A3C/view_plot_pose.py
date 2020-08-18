#!/usr/bin/env python

import rospy

import sys

import time
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64, Int32

def view(x1,y1,x2,y2,x3,y3,x4,y4):
    # plt.plot(int(y1*100),int(x1*100), '*', color="red")
    # plt.plot(int(y2*100),int(x2*100), 'o', color="red")
    # plt.plot(int(y3*100),int(x3*100), '*', color="blue")
    # plt.plot(int(y4*100),int(x4*100), 'o', color="blue")
    line1, = ax.plot(int(y1*100),int(x1*100), '*', color="red")
    line2, = ax.plot(int(y2*100),int(x2*100), 'o', color="red")
    line3, = ax.plot(int(y3*100),int(x3*100), '*', color="blue")
    line4, = ax.plot(int(y4*100),int(x4*100), 'o', color="blue")
    # plt.draw()
    # plt.pause(0.01)
    # plt.clf()
    plt.pause(0.01)
    line1.remove()
    line2.remove()
    line3.remove()
    line4.remove()


if __name__ == '__main__':

    rospy.init_node('view_node')

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-500,500)
    ax.set_ylim(-500,500)
    ax.set_xticks([-500, -200, 0, 200, 500])
    ax.set_yticks([-500, -200, 0, 200, 500])
    #
    # plt.ion()
    # plt.title('HEATMAP')
    # plt.xlim(-500,500)
    # plt.ylim(-500,500)
    # plt.grid()
    # plt.draw()
    # plt.pause(0.01)
    # plt.clf()


    # while True:
    #     time.sleep(1)
    #     view(random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6), random.uniform(-3.6, 3.6), random.uniform(-3.6, 3.6), random.uniform(-3.6, 3.6), random.uniform(-3.6, 3.6))


    # rate = rospy.Rate(3)
    #
    # while not rospy.is_shutdown():
    #
    #     plt.xlim(-500,500)
    #     plt.ylim(-500,500)
    #     plt.grid()
    #     #
    #     # pdata = None
    #     # while pdata is None:
    #     #     try:
    #     #         pdata = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
    #     #     except:
    #     #         pass
    #     #
    #     # ud_pose = pdata.pose[pdata.name.index("ubiquitous_display")]
    #     # x = ud_pose.position.x*100
    #     # y = ud_pose.position.y*100
    #     # plt.plot(int(y),int(x), '*', color="red")
    #     #
    #     # q_x, q_y, q_z, q_w = ud_pose.orientation.x, ud_pose.orientation.y, ud_pose.orientation.z, ud_pose.orientation.w
    #     # ud_yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))
    #     #
    #     # pan_data = None
    #     # while pan_data is None:
    #     #     try:
    #     #         pan_data = rospy.wait_for_message('/ubiquitous_display/pan_conller/command', Float64, timeout=5)
    #     #     except:
    #     #         pass
    #     # tilt_data = None
    #     # while tilt_data is None:
    #     #     try:
    #     #         tilt_data = rospy.wait_for_message('/ubiquitous_display/tilt_conller/command', Float64, timeout=5)
    #     #     except:
    #     #         pass
    #     # radian = math.radians(ud_yaw) + pan_data.data + math.radians(90)
    #     # distance = 0.998 * math.tan(math.radians(90) - tilt_data.data)
    #     # x = (distance * math.cos(radian) + ud_pose.position.x) * 100
    #     # y = (distance * math.sin(radian) + ud_pose.position.y) * 100
    #     # plt.plot(int(y),int(x), 'o', color="red")
    #     #
    #     # actor0_pose = pdata.pose[pdata.name.index("actor0")]
    #     # x = actor0_pose.position.x*100
    #     # y = actor0_pose.position.y*100
    #     # plt.plot(int(y),int(x), '*', color="blue")
    #     #
    #     # q_x, q_y, q_z, q_w = actor0_pose.orientation.x, actor0_pose.orientation.y, actor0_pose.orientation.z, actor0_pose.orientation.w
    #     # actor0_yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))
    #     #
    #     # x = x + (2.5 * math.sin(math.radians(actor0_yaw)))
    #     # y = y - (2.5 * math.cos(math.radians(actor0_yaw)))
    #     # plt.plot(int(y),int(x), 'o', color="blue")
    #     view(random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6),random.uniform(-3.6, 3.6))
    #     rate.sleep()
