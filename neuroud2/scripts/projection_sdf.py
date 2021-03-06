#!/usr/bin/env python

import rospy
import numpy as np
import math
import time
import tf
import os

from std_msgs.msg import String, Bool, Int32, Float64
from geometry_msgs.msg import Point, Vector3, Pose
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates, ModelState

from gazebo_msgs.srv import DeleteModel, DeleteModelRequest, SpawnModel, SpawnModelRequest
import rospkg

MAX_PAN_RADIAN = 2.9670
MIN_PAN_RADIAN = -2.9670

MAX_TILT_RADIAN = 1.3
MIN_TILT_RADIAN = -0.2617

class Projection():
    def __init__(self):

        self.rate = rospy.Rate(5)

        self.x_position = 4.0
        self.y_position = 1.25
        self.offset = 0.5

        self.init_pan_ang = -(math.pi / 2.0)
        self.init_tilt_ang = 0.0

        human_number = 5
        self.name_list = []
        for i in range(human_number):
            self.name_list.append("actor" + str(i))

        self.scan_flag = 0

        self.pan_pub = rospy.Publisher('/ubiquitous_display/pan_controller/command', Float64, queue_size=10)
        self.tilt_pub = rospy.Publisher('/ubiquitous_display/tilt_controller/command', Float64, queue_size=10)

        self.delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        self.image_pub = rospy.Publisher('/ubiquitous_display/image', Int32, queue_size=10)

        path = os.path.join(rospkg.RosPack().get_path('neuroud2')+ '/models')
        with open(path + "/circle_projection.sdf", "r") as f:
            product_xml = f.read()
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            pose = Pose()
            pose.position.x = 0
            pose.position.y = 0
            pose.position.z = -3
            req = SpawnModelRequest()
            req.model_name = "projection"
            req.model_xml = product_xml
            req.initial_pose = pose
            req.reference_frame = "world"
            self.spawn_model(req)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        rospy.sleep(2)
        self.image_pub.publish(0)
        self.pan_pub.publish(-(math.pi / 2.0))
        self.ms_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)
        rospy.loginfo("--- Start projection server ---")

    def service_callback(self, req):
        resp = False

        ud_pose, done_pose = self.get_pose("ubiquitous_display")
        if not done_pose:
            return resp
        ud_ang = self.quaternion_to_euler(ud_pose.orientation)

        for name in self.name_list:
            actor_pose, done_pose = self.get_pose(name)
            if not done_pose:
                return resp
            actor_ang = self.quaternion_to_euler(actor_pose.orientation)

            if actor_ang.z > 0:
                print ("plus")
            elif actor_ang.z < 0 and actor_pose.position.x - ud_pose.position.x > 4.0:
                proj_pos = actor_pose.position.x - 4.0
                # print ("target pose: ", proj_pos, actor_pose.position.y, name)
                # print (int(ud_ang.z))
                distance, radian = self.get_distance(proj_pos, actor_pose.position.y, ud_pose.position.x, ud_pose.position.y,)
                if distance > 1.5 and distance < 2.5:
                    ### calculate pan and tilt radian
                    pan_ang = self.calculate_pan_ang(radian, ud_ang.z)
                    tilt_ang = self.calculate_tilt_ang(distance)

                    ### check pan angle (limit +-2.9670)
                    if abs(pan_ang) < 2.9670:
                        responce = self.set_pantilt_func(pan_ang,tilt_ang)
                        if responce:
                            self.on_off_project(1, proj_pos, actor_pose.position.y)
                            while True:
                                done = self.scan_callback()
                                if done:
                                    break
                                actor_pose, done_pose = self.get_pose(name)
                                if actor_pose.position.x < proj_pos + 1.2 and done_pose == True:
                                    resp = True
                                    break
                                else:
                                    pass
                                # self.rate.sleep()
                            self.on_off_project(0, 0.0, 0.0)

                            # responce = self.set_pantilt_func(-(math.pi / 2.0),0.0)
                        else:
                            pass
            else:
                pass
        responce = self.set_pantilt_func(-(math.pi / 2.0),0.0)
        return resp

    def scan_callback(self):
        done = True
        try:
            data = rospy.wait_for_message('/front_laser_scan', LaserScan, timeout=5)
            state,done = self.calculate_observation(data)
        except:
            pass
        return done

    def calculate_observation(self, data):
        min_range = 0.4
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return data.ranges,done

    def set_pantilt_func(self, pan_ang, tilt_ang):
        response = False
        diff_pan_ang = abs(pan_ang - self.init_pan_ang)
        time_pan = diff_pan_ang * 1.1
        # print diff_pan_ang, time_pan
        diff_tilt_ang = abs(tilt_ang - self.init_tilt_ang)
        time_tilt = diff_tilt_ang * 1.08
        if time_pan > time_tilt:
            target_time = time_pan
        else:
            target_time = time_tilt
        self.pan_pub.publish(pan_ang)
        self.tilt_pub.publish(tilt_ang)

        start = time.time()
        while True:
            elapsed_time = time.time() - start
            if elapsed_time >= target_time:
                response = True
                break
            done = self.scan_callback()
            if done:
                response = False
                break
        return response

    def on_off_project(self, on_off, x, y):
        req = ModelState()
        req.model_name = "projection"
        req.reference_frame = "world"
        if on_off == 1:
            req.pose.position.x = x
            req.pose.position.y = y
            req.pose.position.z = 0
            self.ms_pub.publish(req)

        elif on_off == 0:
            req.pose.position.x = 0
            req.pose.position.y = 0
            req.pose.position.z = -5
            self.ms_pub.publish(req)

    def get_pose(self, name):
        done = False
        responce = Pose()
        try:
            #print "in"
            data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
            #print data.pose[data.name.index(name)]
            done = True
            responce = data.pose[data.name.index(name)]
            #print responce
        except:
            pass
        if done:
            return responce, done
        else:
            return Pose(), done

    def quaternion_to_euler(self, quaternion):
        """Convert Quaternion to Euler Angles

        quarternion: geometry_msgs/Quaternion
        euler: geometry_msgs/Vector3
        """
        e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        return Vector3(x=e[0], y=e[1], z=e[2])

    def get_distance(self, x1, y1, x2, y2):
        d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        r = math.atan2(y1 - y2,x1 - x2)
        return d, r

    def calculate_tilt_ang(self, distance):
        rad_tilt = math.atan2(1.21, distance)
        return rad_tilt

    def calculate_pan_ang(self, radian, ud_ang):
        if radian < 3.14:
            rad_pan = -(math.pi / 2.0) + radian - ud_ang
        else:
            rad_pan = -(math.pi / 2.0) - ((math.pi*2.0)-radian) - ud_ang
        return rad_pan

if __name__ == '__main__':

    rospy.init_node('neuro_server_add_projection')

    server = Projection()
