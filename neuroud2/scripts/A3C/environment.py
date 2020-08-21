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
        self.view_pub = rospy.Publisher('/view', Float64MultiArray, queue_size=10)
        self.past_distance = 0.
        self.past_projector_distance = 0.
        self.yaw = 0
        self.rel_theta = 0
        self.diff_angle = 0
        self.pan_ang = 0.
        self.tilt_ang = 0.
        if is_training:
            self.threshold_arrive = 0.5
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

        view_msg = Float64MultiArray()
        view_msg.data = [self.position.position.x,self.position.position.y,self.projector_position.position.x,self.projector_position.position.y, self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y]
        self.view_pub.publish(view_msg)
        # view_plot_pose.view(self.position.position.x,self.position.position.y,self.projector_position.position.x,self.projector_position.position.y, self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y)

    def getProjState(self):
        reach = False

        radian = math.radians(self.yaw) + self.pan_ang + math.radians(90)
        distance = 0.998 * math.tan(math.radians(90) - self.tilt_ang)
        self.projector_position.position.x = distance * math.cos(radian) + self.position.position.x
        self.projector_position.position.y = distance * math.sin(radian) + self.position.position.y
        diff = math.hypot(self.goal_projector_position.position.x - self.projector_position.position.x, self.goal_projector_position.position.y - self.projector_position.position.y)
        print ("now: ", self.projector_position.position.x, self.projector_position.position.y)
        print ("goal: ", self.goal_projector_position.position.x, self.goal_projector_position.position.y)
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

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_projector_position.position.x - self.position.position.x, self.goal_projector_position.position.y - self.position.position.y)
        # distance_rate = (self.past_distance - current_distance)

        if current_distance <= 3.5:
            distance_rate = current_distance / 3.5
        elif current_distance > 3.5 and current_distance <= 10.0:
            distance_rate = (10 - current_distance) / 6.5
        else:
            distance_rate = 0.
        print ("distance: ", current_distance, distance_rate)

        current_projector_distance, reach = self.getProjState()
        distance_projector_rate = (self.past_projector_distance - current_projector_distance)
        print ("projector: ", current_projector_distance, distance_projector_rate)
        print (arrive, reach)

        cmd_reward = 100.*distance_rate
        proj_reward = 100.* (distance_projector_rate / 4.8600753359177602)
        if proj_reward > 100:
            proj_reward = 100
        elif proj_reward < -100:
            proj_reward = -100
        else:
            proj_reward = proj_reward

        reward = (0.2 * cmd_reward + 0.8 * proj_reward) * 0.01

        self.past_distance = current_distance
        self.past_projector_distance = current_projector_distance
        print ("cmd_reward: ", cmd_reward, "proj_reward: ", proj_reward)
        print ("reward: ", reward)

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
                # self.goal_position.position.x = random.uniform(-4., 4.)
                # self.goal_position.position.y = random.uniform(-4., 4.)
                # self.goal_projector_position.position.x = self.goal_position.position.x - 4.
                # self.goal_projector_position.position.y = self.goal_position.position.y
                self.goal_position.position.x, self.goal_position.position.y, self.goal_projector_position.position.x, self.goal_projector_position.position.y, self.goal_position.orientation = self.cal_actor_pose(2.5)
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.wait_for_service('/gazebo/unpause_physics')
            self.goal_distance = self.getGoalDistace()
            arrive = False
            reach = False

        return reward, reach

    def cal_actor_pose(self, distance):
        xp = 0.
        yp = 0.
        rxp = 0.
        ryp = 0.
        rq = Quaternion()
        while True:
            xp = random.uniform(-3.6, 3.6)
            yp = random.uniform(-3.6, 3.6)
            ang = random.randint(0, 360)
            rxp = xp + (distance * math.sin(math.radians(ang)))
            ryp = yp - (distance * math.cos(math.radians(ang)))
            if abs(rxp) < 3.6 and abs(ryp) < 3.6:
                q = quaternion.from_euler_angles(0,0,math.radians(ang))
                print q
                print type(q)
                rq.x = q.x
                rq.y = q.y
                rq.z = q.z
                rq.w = q.w
                print (xp, " ", yp, " ",ang, " ", rxp, " ",ryp)
                break
        return xp, yp, rxp, ryp, rq


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
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        self.pan_pub.publish(action[2])
        self.tilt_pub.publish(action[3])
        self.pan_ang = action[2]
        self.tilt_ang = action[3]

        time.sleep(0.5)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front_laser_scan', LaserScan, timeout=5)
            except:
                pass
        # pdata = None
        # while pdata is None:
        #     try:
        #         pdata = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
        #     except:
        #         pass
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
        reward, reach = self.setReward(done, arrive)

        return np.asarray(state), reward, done, arrive, reach

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
            # self.goal_position.position.x = random.uniform(-4., 4.)
            # self.goal_position.position.y = random.uniform(-4., 4.)
            # self.goal_projector_position.position.x = self.goal_position.position.x - 4.
            # self.goal_projector_position.position.y = self.goal_position.position.y
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
