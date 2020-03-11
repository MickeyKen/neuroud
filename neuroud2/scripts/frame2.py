#!/usr/bin/env python
import numpy as np
import rospy
import time
import random
import time
# import liveplot
import my_qlearn
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import numpy
from gym.spaces import *

import matplotlib.pyplot as plt

# class Subscribe():
#     def __init__(self):


def discretize_observation(data,new_ranges):
    discretized_ranges = []
    min_range = 0.4
    done = False
    mod = len(data.ranges)/new_ranges
    for i, item in enumerate(data.ranges):
        if (i%mod==0):
            if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                discretized_ranges.append(6)
            elif np.isnan(data.ranges[i]):
                discretized_ranges.append(0)
            else:
                discretized_ranges.append(int(data.ranges[i]))
        if (min_range > data.ranges[i] > 0):
            done = True
    return discretized_ranges,done

def min_pooling(data,new_ranges):
    discretized_ranges = []
    min_range = 0.4
    done = False
    mod = len(data.ranges)/new_ranges

    ### scan range 1,083 ###
    for i, item in enumerate(data.ranges):
        if (i%mod==0 and i+mod < 1084):
            minData = min(data.ranges[i:i+mod])
            # print minData, int(minData), i+mod
            if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                discretized_ranges.append(6)
            elif np.isnan(data.ranges[i]):
                discretized_ranges.append(0)
            else:
                discretized_ranges.append(int(minData))
        if (min_range > minData > 0):
            done = True
    return discretized_ranges,done

def seed(seed=None):
    np_random, seed = seeding.np_random(seed)
    return [seed]

def step(action):

    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        unpause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/unpause_physics service call failed")

    if action == 0: #FORWARD
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.3
        vel_cmd.angular.z = 0.0
        vel_pub.publish(vel_cmd)
    elif action == 1: #LEFT
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.05
        vel_cmd.angular.z = 0.3
        vel_pub.publish(vel_cmd)
    elif action == 2: #RIGHT
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.05
        vel_cmd.angular.z = -0.3
        vel_pub.publish(vel_cmd)

    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front_laser_scan', LaserScan, timeout=5)
        except:
            pass

    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        #resp_pause = pause.call()
        pause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/pause_physics service call failed")

    state,done = discretize_observation(data,5)

    if not done:
        if action == 0:
            reward = 5
        else:
            reward = 1
    else:
        reward = -2000

    return state, reward, done, {}

def reset():

    # Resets the state of the environment and returns an initial observation.
    rospy.wait_for_service('/gazebo/reset_simulation')
    try:
        #reset_proxy.call()
        reset_proxy()
    except (rospy.ServiceException) as e:
        print ("/gazebo/reset_simulation service call failed")

    # Unpause simulation to make observation
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        #resp_pause = pause.call()
        unpause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/unpause_physics service call failed")

    #read laser data
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/front_laser_scan', LaserScan, timeout=5)
        except:
            pass


    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        #resp_pause = pause.call()
        pause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/pause_physics service call failed")

    state = discretize_observation(data,5)

    return state

if __name__ == '__main__':

    rospy.init_node('tcmdvel_publisher')

    vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)


    last_time_steps = numpy.ndarray(0)

    qlearn = my_qlearn.MyQLearn([0,1,2],
                    alpha=0.2, gamma=0.8, epsilon=0.6)

    initial_epsilon = qlearn.epsilon

    epsilon_discount = 0.9986

    start_time = time.time()
    total_episodes = 10000
    highest_reward = 0

    plt.ion()
    plt.title('Simple Curve Graph')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.xlim(0,1000)
    plt.ylim(-2000,5000)
    plt.grid()
    xx = []
    y = []
    y2 = []


    for x in range(total_episodes):
        done = False

        cumulated_reward = 0 #Should going forward give more reward then L/R ?

        observation = reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        #render() #definednvidia-detector devices above, not env.render()

        state = ''.join(map(str, observation))

        for i in range(1500):

            # print qlearn.q

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = step(action)
            # print observation
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))
            # print nextState

            # print "====================================="

            qlearn.learn(state, action, reward, nextState)
            # print state

            # print ("Q value: " ,qlearn.q)

            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
        xx.append(x+1)
        y.append(cumulated_reward)
        plt.plot(xx,y, color="blue")
        y2.append(cumulated_reward/i)
        plt.plot(xx,y2, color="red")
        plt.draw()
        plt.pause(0.1)
        # m, s = divmod(int(time.time() - start_time), 60)
        # h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+" - Count: "+str(i))
    plt.close()
