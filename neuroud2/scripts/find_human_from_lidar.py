#!/usr/bin/python
import rospy
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates

import numpy as np
import matplotlib.pyplot as plt
import math

def find_human(data, md_data):
    ud_x = md_data.pose[md_data.name.index("ubiquitous_display")].position.x
    discretized_ranges = []
    x_r = []
    y_r = []
    min_range = 0.2
    print len(data.ranges)
    step = math.pi / len(data.ranges)
    human_count = 0
    Human = False
    for i, item in enumerate(data.ranges):
        distance = data.ranges[i]
        x = distance * math.cos(step * (i+1)) + ud_x
        y = distance * math.sin(step * (i+1))
        # print step * (i+1)
        if x < 3.4 and x > -3.4 and y < 5.4 and y > 2.6:
            human_count += 1
        x_r.append(x)
        y_r.append(y)
    if human_count > 8:
        Human = True

    return x_r, y_r, Human

if __name__ == '__main__':
    rospy.init_node('analysis_lidar_node', anonymous=True)

    while True:
        plt.ion()
        plt.title('HEATMAP')
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.grid()
        data = rospy.wait_for_message('/scan_filtered', LaserScan, timeout=5)
        md_data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)

    	x_value, y_value, h = find_human(data,md_data)
        print h

        plt.plot(x_value,y_value, 'o', color="blue")
        plt.draw()
        plt.pause(0.1)
        plt.clf() 


    # rospy.spin()
