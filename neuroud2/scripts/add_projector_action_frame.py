#!/usr/bin/env python

import gym
from gym import wrappers
# import gym_gazebo
import time
import numpy as np
from numpy import inf
# from distutils.dir_util import copy_tree
import os
import json
# import liveplot
import my_dqn

import rospy

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan

from gym.utils import seeding

import matplotlib.pyplot as plt

import projection_topic



def calculate_observation(data):
    min_range = 0.6
    done = False
    for i, item in enumerate(data.ranges):
        if (min_range > data.ranges[i] > 0):
            done = True
    return data.ranges,done

def seed(seed=None):
    np_random, seed = seeding.np_random(seed)
    return [seed]

def step(action, last_action):
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        unpause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/unpause_physics service call failed")

    if action == 0: #STOP
        vel_cmd = Twist()
        vel_pub.publish(vel_cmd)

    elif action == 1: #LEFT
        vel_cmd = Twist()
        vel_cmd.linear.y = 1.0
        vel_pub.publish(vel_cmd)
    elif action == 2: #RIGHT
        vel_cmd = Twist()
        vel_cmd.linear.y = -1.0
        vel_pub.publish(vel_cmd)

    elif action == 3:
        if last_action == 0:
            response = p.service_callback(False)
            if response:
                action = 33

    time.sleep(0.5)

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

    state,done = calculate_observation(data)

    if not done:
        if action == 0 or action == 3:
            reward = 2
        elif action == 33:
            reward = 20
        else:
            reward = -1
    else:
        reward = -200
    print reward

    state = np.asarray(state)
    state[np.isnan(state)] = 0.5
    state[np.isinf(state)] = 30.0
    # np.nan_to_num(state)
    state[state == inf] = 30.0
    state[state == -inf] = 30.0
    # print state


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

    state,done = calculate_observation(data)
    state = np.asarray(state)
    state[np.isnan(state)] = 0.5
    state[np.isinf(state)] = 30.0
    # np.nan_to_num(state)
    state[state == inf] = 30.0
    state[state == -inf] = 30.0
    return state

def plot(x1,y1,x2,y2,cumulated_reward):
    xx.append(epoch+1)
    y.append(cumulated_reward)
    plt.plot(xx,y, color="blue")
    plt.draw()
    plt.pause(0.1)

if __name__ == '__main__':
    rospy.init_node('origin_tcmdvel_publisher_dqn')

    vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    p = projection_topic.Projection()

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    # env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    path = '/tmp/'
    # plotter = liveplot.LivePlot(outdir)

    continue_execution = False

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 1000
        steps = 1000
        updateTargetNetwork = 10000
        explorationRate = 0.6
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 1080
        network_outputs = 4

        ### number of hiddenLayer ###
        network_structure = [300,21]
        current_epoch = 0

        deepQ = my_dqn.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
    else:
        pass

    # env._max_episode_steps = steps # env returns done after _max_episode_steps
    # env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    plt.ion()
    plt.title('Simple Curve Graph')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.xlim(0,5000)
    plt.ylim(-500,3000)
    plt.grid()
    xx = []
    y = []
    y2 = []

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = reset()
        cumulated_reward = 0
        done = False
        episode_step = 0
        last_action = 0

        # run until env returns done
        while not done:
            # env.render()
            qValues = deepQ.getQValues(observation)
            print ("ss" ,qValues)

            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = step(action, last_action)
            last_action = action

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, newObservation, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                    # print "pass False"
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)
                    # print "pass True"

            observation = newObservation

            if done:
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))
                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%100==0:
                        deepQ.saveModel(path+str(epoch)+'.h5')

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

            episode_step += 1

        plot(xx,y,xx,y2,cumulated_reward)

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

    plt.savefig('output.png')
