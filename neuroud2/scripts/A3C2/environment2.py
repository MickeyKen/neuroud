#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random
import quaternion
import time

from std_msgs.msg import Float64, Int32, Float64MultiArray
from geometry_msgs.msg import Twist, Point, Pose, Vector3, Quaternion
from sensor_msgs.msg import LaserScan, JointState
# from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState, SetModelStateRequest

goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..',  '..','neuroud2'
                                , 'models', 'person_standing', 'model.sdf')

PAN_LIMIT = math.radians(180)  #2.9670
TILT_MIN_LIMIT = math.radians(90) - math.atan(3.0/0.998)
TILT_MAX_LIMIT = math.radians(90) - math.atan(1.5/0.998)

class Env():
    def __init__(self, is_training):
        self.position = Pose()
        self.projector_position = Pose()
        self.goal_position = Pose()
        self.goal_projector_position = Pose()

        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.ud_spawn = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.pan_pub = rospy.Publisher('/ubiquitous_display/pan_controller/command', Float64, queue_size=10)
        self.tilt_pub = rospy.Publisher('/ubiquitous_display/tilt_controller/command', Float64, queue_size=10)
        self.sub_jsp = rospy.Subscriber('/ubiquitous_display/joint_states', JointState, self.getJsp)

        self.yaw = 0
        self.pan_ang = 0.
        self.tilt_ang = 0.

        self.view_pub = rospy.Publisher('/view', Float64MultiArray, queue_size=10)

        if is_training:
            self.threshold_reach = 0.25
        else:
            self.threshold_reach = 0.5

    def constrain(self, input, low, high):
        if input < low:
          input = low
        elif input > high:
          input = high
        else:
          input = input

        return input

    def getJsp(self, jsp):
        self.pan_ang = jsp.position[jsp.name.index("pan_joint")]
        self.tilt_ang = jsp.position[jsp.name.index("tilt_joint")]

    def pubView(self):
        view_msg = Float64MultiArray()
        view_msg.data = [self.position.position.x,self.position.position.y,self.projector_position.position.x,self.projector_position.position.y, self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y]
        self.view_pub.publish(view_msg)

    def cal_actor_pose(self, distance):
        xp = 0.
        yp = 0.
        rxp = 0.
        ryp = 0.
        while True:
            xp = random.uniform(-3.4, 3.4)
            yp = random.uniform(-3.4, 3.4)
            ang = random.randint(0, 360)
            rxp = xp + (distance * math.sin(math.radians(ang)))
            ryp = yp - (distance * math.cos(math.radians(ang)))
            if rxp >= -3.4 and rxp <= 3.4 and ryp >= -3.4 and ryp <= 3.4:
                q = quaternion.from_euler_angles(0,0,math.radians(ang))
                break
        rq = Quaternion()
        rq.x = q.x
        rq.y = q.y
        rq.z = q.z
        rq.w = q.w
        return xp, yp, rxp, ryp, rq

    def udPosition(self):
        p_x = self.goal_projector_position.position.x
        p_y = self.goal_projector_position.position.y

        while True:
            distance = random.uniform(1.5, 3.0)
            ang = random.randint(0, 360)
            ud_x = p_x + (distance * math.sin(math.radians(ang)))
            ud_y = p_y - (distance * math.cos(math.radians(ang)))
            if ud_x >= -3.4 and ud_x <= 3.4 and ud_y >= -3.4 and ud_y <= 3.4:
                diff = math.hypot(self.goal_position.position.x - ud_x, self.goal_position.position.y - ud_y)
                if diff > 0.4 + 0.271 + 0.3:
                    ud_ang = random.randint(0, 360)
                    self.yaw = math.radians(ud_ang)
                    q = quaternion.from_euler_angles(0,0,math.radians(ud_ang))
                    break

        self.position.position.x = ud_x
        self.position.position.y = ud_y
        rq = Quaternion()
        rq.x = q.x
        rq.y = q.y
        rq.z = q.z
        rq.w = q.w
        return ud_x, ud_y, rq

    def getState(self, scan):
        scan_range = []
        min_range = 0.4 + 0.271

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(30.)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        return scan_range

    def getProjState(self):
        reach = False

        radian = self.yaw + self.pan_ang + math.radians(90)
        distance = 0.998 * math.tan(math.radians(90) - self.tilt_ang)
        self.projector_position.position.x = distance * math.cos(radian) + self.position.position.x
        self.projector_position.position.y = distance * math.sin(radian) + self.position.position.y
        diff = math.hypot(self.goal_projector_position.position.x - self.projector_position.position.x, self.goal_projector_position.position.y - self.projector_position.position.y)

        # diff_ang = self.pan_ang + self.yaw
        # diff_dis = self.tilt

        if diff <= self.threshold_reach:
            reach = True

        return diff, reach

    def setReward(self, current_distance, reach):

        if reach:
            reward = 200

            rospy.wait_for_service('/gazebo/delete_model')
            try:
                self.del_model('actor0')
            except rospy.ServiceException, e:
                print ("Service call failed: %s" % e)

            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'actor0'  # the same with sdf name
                target.model_xml = goal_urdf
                self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y, self.goal_position.orientation = self.cal_actor_pose(2.5)
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")

            x,y,orientation = self.udPosition()

            # Build the targetz
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                srv = SetModelStateRequest()
                srv.model_state.model_name = 'ubiquitous_display'
                srv.model_state.reference_frame = 'world'  # the same with sdf name
                srv.model_state.pose.position.x = x
                srv.model_state.pose.position.y = y
                srv.model_state.pose.orientation = orientation
                self.ud_spawn(srv)
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")

        else:
            reward = 1 - (((self.threshold_reach - current_distance) ** 2) / 2.)*0.6

        return reward


    def step(self, action, past_action):

        pan_ang = self.constrain(self.pan_ang + action[0], -PAN_LIMIT, PAN_LIMIT)
        tilt_ang = self.constrain(self.tilt_ang + action[1], TILT_MIN_LIMIT, TILT_MAX_LIMIT)
        self.pan_pub.publish(pan_ang)
        self.tilt_pub.publish(tilt_ang)

        time.sleep(0.5)

        self.pubView()

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state = self.getState(data)

        state = [i / 30. for i in state]

        for pa in past_action:
            state.append(pa)

        state.append(self.pan_ang)
        state.append(self.tilt_ang)

        # state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        current_distance, reach = self.getProjState()
        reward = self.setReward(current_distance, reach)

        return np.asarray(state), reward, reach

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.del_model('actor0')
        except rospy.ServiceException, e:
            print ("Service call failed: %s" % e)

        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_world service call failed")

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'actor0'  # the same with sdf name
            target.model_xml = goal_urdf
            self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y, self.goal_position.orientation = self.cal_actor_pose(2.5)
            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")

        x,y,orientation = self.udPosition()

        # Build the targetz
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            srv = SetModelStateRequest()
            srv.model_state.model_name = 'ubiquitous_display'
            srv.model_state.reference_frame = 'world'  # the same with sdf name
            srv.model_state.pose.position.x = x
            srv.model_state.pose.position.y = y
            srv.model_state.pose.orientation = orientation
            self.ud_spawn(srv)
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state = self.getState(data)
        state = [i / 30. for i in state]

        self.pan_pub.publish(0.)
        self.tilt_pub.publish(TILT_MIN_LIMIT)

        state.append(0)
        state.append(TILT_MIN_LIMIT)
        state.append(0)
        state.append(TILT_MIN_LIMIT)
        # state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)
