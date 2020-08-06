#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random

from std_msgs.msg import Float64, Int32
from geometry_msgs.msg import Twist, Point, Pose, Vector3
from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
# from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates

class Env():
    def __init__(self, is_training):
        self.actor_num = 5
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.pan_pub = rospy.Publisher('/ubiquitous_display/pan_controller/command', Float64, queue_size=10)
        self.tilt_pub = rospy.Publisher('/ubiquitous_display/tilt_controller/command', Float64, queue_size=10)
        self.image_pub = rospy.Publisher('/ubiquitous_display/image', Int32, queue_size=10)

    def getState(self, scan, action):
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

        reach = self.calculate_point(action)

        return scan_range, done, reach

    def setReward(self, done, reach, action):

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

    def calculate_point(self, action):
        try:
            data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
        except:
            pass
        ud_pose = data.pose[data.name.index("ubiquitous_display")]
        ud_ang = self.quaternion_to_euler(ud_pose.orientation)
        radian = ud_ang.z + action[2] - math.radians(90)
        distance = math.tan(action[3]) * 0.998
        x_pos = distance * math.cos(radian)
        y_pos = distance * math.sin(radian)

        for i in data:
            if i.name != "ubiquitous_display":
                if i.position.y > y_pos - 0.2 and i.position < y_pos + 0.2:
                    if i.position.x - 2.5 > x_pos - 0.2 and i.position.x - 2.5 < x_pos + 0.2:
                        return True
        return False

    def quaternion_to_euler(self, quaternion):
        """Convert Quaternion to Euler Angles

        quarternion: geometry_msgs/Quaternion
        euler: geometry_msgs/Vector3
        """
        e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return Vector3(x=e[0], y=e[1], z=e[2])

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        self.pan_pub.publish(action[2])
        self.tilt_pub.publish(action[3])

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front_laser_scan', LaserScan, timeout=5)
            except:
                pass

        state, done, reach = self.getState(data, action)
        state = [i / 30. for i in state]

        # for pa in past_action:
        #     state.append(pa)

        # state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, arrive, reach, action)

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
