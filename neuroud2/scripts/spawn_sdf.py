#!/usr/bin/env python

import rospy, tf
import os
from gazebo_msgs.srv import DeleteModel, SpawnModel, SpawnModelRequest
from geometry_msgs.msg import *
import rospkg

if __name__ == '__main__':
    print("Waiting for gazebo services...")
    rospy.init_node("spawn_products_in_bins")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    print("Got it.")
    path = os.path.join(rospkg.RosPack().get_path('neuroud2')+ '/models')
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    with open(path + "/circle_projection.sdf", "r") as f:
        product_xml = f.read()

    pose = Pose()
    pose.position.y = 0.0
    req = SpawnModelRequest()
    req.model_name = "block"
    req.model_xml = product_xml
    req.initial_pose = pose
    req.reference_frame = "world"
    # print req

    spawn_model(req)
    #
    # for num in xrange(0,12):
    #     item_name = "product_{0}_0".format(num)
    #     print("Deleting model:%s", item_name)
    #     delete_model(item_name)
    #
    # for num in xrange(0,12):
    #     bin_y   =   2.8 *   (num    /   6)  -   1.4
    #     bin_x   =   0.5 *   (num    %   6)  -   1.5
    #     item_name   =   "product_{0}_0".format(num)
    #     print("Spawning model:%s", item_name)
    #     item_pose   =   Pose(Point(x=bin_x, y=bin_y,    z=2),   orient)
    #     spawn_model(item_name, product_xml, "", item_pose, "world")
