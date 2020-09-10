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


diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..','neuroud2'
                                , 'models', 'person_standing', 'model.sdf')

PAN_LIMIT = math.radians(90)  #2.9670
TILT_MIN_LIMIT = math.radians(90) - math.atan(3.0/0.998)
TILT_MAX_LIMIT = math.radians(90) - math.atan(1.5/0.998)

class Env():
    def __init__(self, is_training):
        self.position = Pose()
        self.position.position.x = 0.
        self.position.position.y = 0.
        self.projector_position = Pose()
        self.projector_position.position.x = 0.
        self.projector_position.position.y = 0.
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.goal_projector_position = Pose()
        self.goal_projector_position.position.x = 0.
        self.goal_projector_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('/gazebo/model_states', ModelStates, self.getPose)
        self.sub_jsp = rospy.Subscriber('/ubiquitous_display/joint_states', JointState, self.getJsp)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.pan_pub = rospy.Publisher('/ubiquitous_display/pan_controller/command', Float64, queue_size=10)
        self.tilt_pub = rospy.Publisher('/ubiquitous_display/tilt_controller/command', Float64, queue_size=10)
        self.image_pub = rospy.Publisher('/ubiquitous_display/image', Int32, queue_size=10)
        self.view_pub = rospy.Publisher('/view', Float64MultiArray, queue_size=10)
        self.past_distance = 0.
        self.past_distance_rate = 0.
        self.past_projector_distance = 0.
        self.yaw = 0
        self.rel_theta = 0
        self.diff_angle = 0
        self.pan_ang = 0.
        self.tilt_ang = 0.
        self.v = 0.
        if is_training:
            self.threshold_arrive = 0.25
            self.min_threshold_arrive = 1.5
            self.max_threshold_arrive = 3.0
        else:
            self.threshold_arrive = 0.5
            self.min_threshold_arrive = 1.5
            self.max_threshold_arrive = 3.0

    def constrain(self, input, low, high):
        if input < low:
          input = low
        elif input > high:
          input = high
        else:
          input = input

        return input

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.position.x, self.goal_position.position.y - self.position.position.y)
        self.past_distance = goal_distance

        return goal_distance

    def getJsp(self, jsp):
        self.pan_ang = jsp.position[jsp.name.index("pan_joint")]
        self.tilt_ang = jsp.position[jsp.name.index("tilt_joint")]

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

        # view_msg = Float64MultiArray()
        # view_msg.data = [self.position.position.x,self.position.position.y,self.projector_position.position.x,self.projector_position.position.y, self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y]
        # self.view_pub.publish(view_msg)
        # view_plot_pose.view(self.position.position.x,self.position.position.y,self.projector_position.position.x,self.projector_position.position.y, self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y)

    def getProjState(self):
        reach = False

        radian = math.radians(self.yaw) + self.pan_ang + math.radians(90)
        distance = 0.998 * math.tan(math.radians(90) - self.tilt_ang)
        self.projector_position.position.x = distance * math.cos(radian) + self.position.position.x
        self.projector_position.position.y = distance * math.sin(radian) + self.position.position.y
        diff = math.hypot(self.goal_projector_position.position.x - self.projector_position.position.x, self.goal_projector_position.position.y - self.projector_position.position.y)
        # print ("now: ", self.projector_position.position.x, self.projector_position.position.y)
        # print ("goal: ", self.goal_projector_position.position.x, self.goal_projector_position.position.y)
        if diff <= self.threshold_arrive:
            # done = True
            reach = True
        return diff, reach

    def getState(self, scan):
        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.4 + 0.271
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

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_projector_position.position.x - self.position.position.x, self.goal_projector_position.position.y - self.position.position.y)
        # distance_rate = (self.past_distance - current_distance)

        if current_distance >= 2.25:
            r_c = ((2.25 - current_distance) ** 2) / 2
        else:
            r_c = 1 - (current_distance / 2.25)

        # print ("distance: ", current_distance, distance_rate)
        print (current_distance, r_c)

        current_projector_distance, reach = self.getProjState()
        # distance_projector_rate = (self.past_projector_distance - current_projector_distance)
        if current_projector_distance >= 0.25:
            r_p = ((0.25 - current_projector_distance) ** 2) / 2
        else:
            r_p = 0
        print (current_projector_distance, r_p)

        # cmd_reward = 100.* distance_rate
        # proj_reward = 100. * projector_distance_rate
        # proj_reward = 100.* (distance_projector_rate / 1.1360454260284)
        # proj_reward = self.constrain(proj_reward, -100, 100)

        # reward = (0.4 * cmd_reward + 0.6 * proj_reward) * 0.01
        reward = 1 - r_c - r_p - self.v
        #
        # self.past_distance = current_distance
        # self.past_distance_rate = distance_rate
        # self.past_projector_distance = current_projector_distance
        print ("cmd_reward: ", round(r_c,2), "proj_reward: ", round(r_p,2), "total_reward", round(reward,2))

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if arrive == True and reach == True:
            reward = 200.

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

            self.goal_distance = self.getGoalDistace()
            # arrive = False
            # reach = False
        # print ("reward: ", reward)

        return reward, reach


    def step(self, action, past_action):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        vel_cmd = Twist()
        if action == 0:
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 1:
            vel_cmd.linear.x = 0.1
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 2:
            vel_cmd.linear.x = -0.1
            self.pub_cmd_vel.publish(vel_cmd)
        elif action == 3:
            pan_ang = self.constrain(self.pan_ang + 0.15, -PAN_LIMIT, PAN_LIMIT)
            self.pan_pub.publish(pan_ang)
        elif action == 4:
            pan_ang = self.constrain(self.pan_ang - 0.15, -PAN_LIMIT, PAN_LIMIT)
            self.pan_pub.publish(pan_ang)
        elif action == 5:
            tilt_ang = self.constrain(self.tilt_ang + 0.08, TILT_MIN_LIMIT, TILT_MAX_LIMIT)
            self.tilt_pub.publish(tilt_ang)
        elif action == 6:
            tilt_ang = self.constrain(self.tilt_ang - 0.08, TILT_MIN_LIMIT, TILT_MAX_LIMIT)
            self.tilt_pub.publish(tilt_ang)
        else:
            print ("Error action is from 0 to 6")

        time.sleep(0.5)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan_filtered', LaserScan, timeout=5)
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

        state.append(past_action)
        state.append(self.pan_ang)
        state.append(self.tilt_ang)

        # state = state + [yaw / 360, rel_theta / 360, diff_angle / 180]
        reward, reach = self.setReward(done, arrive)

        return np.asarray(state), reward, done

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

    def reset(self):
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


        # Build the targetz
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

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan_filtered', LaserScan, timeout=5)
            except:
                pass

        self.pan_ang = 0.
        self.tilt_ang = TILT_MIN_LIMIT
        self.pan_pub.publish(self.pan_ang)
        self.tilt_pub.publish(self.tilt_ang)

        self.past_distance_rate, reach = self.getProjState()

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 30. for i in state]

        self.pan_pub.publish(0.)
        self.tilt_pub.publish(TILT_MIN_LIMIT)

        state.append(0)
        state.append(0)
        state.append(TILT_MIN_LIMIT)

        # state = state + [yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)
