#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
# from gazebo_msgs.srv import SpawnModel, DeleteModel

class Env():
    def __init__(self, is_training):
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)


    def getState(self, scan):
        scan_range = []
        min_range = 0.4
        done = False
        arrive = False

        for i in range(len(scan.ranges)):
            if np.isinf(scan.ranges[i]):
                scan_range.append(30)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        return scan_range, done

    def setReward(self, done):

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        return reward

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front_laser_scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        state = [i / 30. for i in state]

        # for pa in past_action:
        #     state.append(pa)

        # state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, arrive)

        return np.asarray(state), reward, done

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('target')

        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_world service call failed")

        rospy.wait_for_service('/gazebo/unpause_physics')
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front_laser_scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        state = [i / 30. for i in state]

        state.append(0)
        state.append(0)

        # state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)
