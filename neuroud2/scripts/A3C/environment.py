#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random
import quaternion
import time

from std_msgs.msg import Float64, Int32
from geometry_msgs.msg import Twist, Point, Pose, Vector3
from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..',  '..','neuroud2'
                                , 'models', 'person_standing', 'model.sdf')


class Env():
    def __init__(self, is_training):
        self.position = Pose()
        self.projector_position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.goal_projector_position = Pose()
        self.goal_projector_position.position.x = 0.
        self.goal_projector_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('/gazebo/model_states', ModelStates, self.getPose)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.pan_pub = rospy.Publisher('/ubiquitous_display/pan_controller/command', Float64, queue_size=10)
        self.tilt_pub = rospy.Publisher('/ubiquitous_display/tilt_controller/command', Float64, queue_size=10)
        self.image_pub = rospy.Publisher('/ubiquitous_display/image', Int32, queue_size=10)
        self.past_distance = 0.
        self.past_projector_distance = 0.
        self.yaw = 0
        self.rel_theta = 0
        self.diff_angle = 0
        if is_training:
            self.threshold_arrive = 0.2
            self.min_threshold_arrive = 1.5
            self.max_threshold_arrive = 5.0
        else:
            self.threshold_arrive = 0.4

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.position.x, self.goal_position.position.y - self.position.position.y)
        self.past_distance = goal_distance

        return goal_distance

    def getPose(self, pose):
        self.position = pose.pose[pose.name.index("ubiquitous_display")]
        orientation = self.position.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_projector_position.position.x - self.position.position.x, 1)
        rel_dis_y = round(self.goal_projector_position.position.y - self.position.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getProjState(self, action):
        reach = False

        pan_rad = action[2]
        tilt_rad = action[3]
        radian = math.radians(self.yaw) + pan_rad - math.radians(90)
        distance = 0.998 * math.tan(tilt_rad)
        self.projector_position.position.x = distance * math.cos(radian) + self.position.position.x
        self.projector_position.position.y = distance * math.sin(radian) + self.position.position.y
        diff = math.hypot(self.goal_projector_position.position.x - self.projector_position.position.x, self.goal_projector_position.position.y - self.projector_position.position.y)
        if diff <= self.threshold_arrive:
            # done = True
            reach = True
        return diff, reach

    def getState(self, scan):
        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.4
        done = False
        arrive = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(30.)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = math.hypot(self.goal_projector_position.position.x- self.position.position.x, self.goal_projector_position.position.y - self.position.position.y)
        if current_distance >= self.min_threshold_arrive and current_distance <= self.max_threshold_arrive:
            # done = True
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive, action):
        current_distance = math.hypot(self.goal_projector_position.position.x - self.position.position.x, self.goal_projector_position.position.y - self.position.position.y)
        distance_rate = (self.past_distance - current_distance)

        current_projector_distance, reach = self.getProjState(action)
        distance_projector_rate = (self.past_projector_distance - current_projector_distance)

        reward = 250.*distance_rate + 250.*distance_projector_rate
        self.past_distance = current_distance
        self.past_projector_distance = current_projector_distance

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if arrive == True and reach == True:
            reward = 120.
            self.pub_cmd_vel.publish(Twist())
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('actor0')

            # Build the target
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'actor0'  # the same with sdf name
                target.model_xml = goal_urdf
                self.goal_position.position.x = random.uniform(-4., 4.)
                self.goal_position.position.y = random.uniform(-4., 4.)
                self.goal_projector_position.position.x = self.goal_position.position.x - 4.
                self.goal_projector_position.position.y = self.goal_position.position.y
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.wait_for_service('/gazebo/unpause_physics')
            self.goal_distance = self.getGoalDistace()
            arrive = False
            reach = False

        return reward, reach

    def step(self, action, past_action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.linear.y = ang_vel / 4
        self.pub_cmd_vel.publish(vel_cmd)
        self.pan_pub.publish(action[2])
        self.tilt_pub.publish(action[3])

        time.sleep(0.5)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front_laser_scan', LaserScan, timeout=5)
            except:
                pass
        pdata = None
        while pdata is None:
            try:
                pdata = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
            except:
                pass
        rospy.wait_for_service('/gazebo/pause_physics')

        try:
            #resp_pause = pause.call()
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 30. for i in state]

        for pa in past_action:
            state.append(pa)

        state = state + [yaw / 360, rel_theta / 360, diff_angle / 180]
        reward, reach = self.setReward(done, arrive, action)

        return np.asarray(state), reward, done, arrive, reach

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('actor0')

        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_world service call failed")

        # Build the targetz
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'actor0'  # the same with sdf name
            target.model_xml = goal_urdf
            self.goal_position.position.x = random.uniform(-4., 4.)
            self.goal_position.position.y = random.uniform(-4., 4.)
            self.goal_projector_position.position.x = self.goal_position.position.x - 4.
            self.goal_projector_position.position.y = self.goal_position.position.y
            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
            
        rospy.wait_for_service('/gazebo/unpause_physics')
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front_laser_scan', LaserScan, timeout=5)
            except:
                pass

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 30. for i in state]

        state.append(0)
        state.append(0)
        state.append(0)
        state.append(0)

        state = state + [yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)
