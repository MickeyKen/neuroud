#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random
import quaternion

from std_msgs.msg import Float64, Int32
from geometry_msgs.msg import Twist, Point, Pose, Vector3
from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
# from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates

class Env():
    def __init__(self, is_training):
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.pan_pub = rospy.Publisher('/ubiquitous_display/pan_controller/command', Float64, queue_size=10)
        self.tilt_pub = rospy.Publisher('/ubiquitous_display/tilt_controller/command', Float64, queue_size=10)
        self.image_pub = rospy.Publisher('/ubiquitous_display/image', Int32, queue_size=10)

    def cal_actor_pose(self, distance):
        xp = 0.
        yp = 0.
        rxp = 0.
        ryp = 0.
        rq = Quaternion()
        xp = random.uniform(-3.0, 3.0)
        yp = random.uniform(3.0, 5.0)
        ang = 0
        rxp = xp + (distance * math.sin(math.radians(ang)))
        ryp = yp - (distance * math.cos(math.radians(ang)))
        q = quaternion.from_euler_angles(0,0,math.radians(ang))
        rq.x = q.x
        rq.y = q.y
        rq.z = q.z
        rq.w = q.w
        return xp, yp, rxp, ryp, rq

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

    def setReward(self, done, reach, action):
        reward = 5

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if reach:
            reward = 30
            if action[0] == 0.0 and action[1] == 0.0:
                reward = 60
                self.image_pub.publish(1)
            else:
                self.image_pub.publish(0)
        else:
            self.image_pub.publish(0)


        return reward

    def calculate_point(self, pdata, action):
        ud_pose = pdata.pose[pdata.name.index("ubiquitous_display")]
        ud_ang = self.quaternion_to_euler(ud_pose.orientation)
        radian = ud_ang + action[2] - math.radians(90)
        distance = math.tan(action[3]) * 0.998
        x_pos = distance * math.cos(radian)
        y_pos = distance * math.sin(radian)
        for i in range(self.actor_num):
            actor_pos = pdata.pose[pdata.name.index("actor" + str(i))]
            if actor_pos.position.y > y_pos - 0.2 and actor_pos.position < y_pos + 0.2:
                if actor_pos.position.x - 2.5 > x_pos - 0.2 and actor_pos.position.x - 2.5 < x_pos + 0.2:
                    return True
        return False

    def quaternion_to_euler(self, q):
        # print (q.z)

        quat = np.quaternion(q.w, q.x, q.y, q.z)
        euler = quaternion.as_euler_angles(quat)
        return euler[0]


    def step(self, action, past_action):

        self.pan_pub.publish(action[0])
        self.tilt_pub.publish(action[1])

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front_laser_scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reach = self.calculate_point(pdata, action)
        state = [i / 30. for i in state]

        for pa in past_action:
            state.append(pa)

        # state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, reach, action)

        return np.asarray(state), reward, done

    def reset(self):
        # Reset the env #

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
