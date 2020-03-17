#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
from gym import wrappers
# import gym_gazebo
import time
import numpy as np
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
# def detect_monitor_files(training_dir):
#     return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]
#
# def clear_monitor_files(training_dir):
#     files = detect_monitor_files(training_dir)
#     if len(files) == 0:
#         return
#     for file in files:
#         print(file)
#         os.unlink(file)
def calculate_observation(data):
    min_range = 0.2
    done = False
    for i, item in enumerate(data.ranges):
        if (min_range > data.ranges[i] > 0):
            done = True
    return data.ranges,done

def seed(seed=None):
    np_random, seed = seeding.np_random(seed)
    return [seed]

def step(action):
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        unpause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/unpause_physics service call failed")

    max_ang_speed = 0.3
    ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

    vel_cmd = Twist()
    vel_cmd.linear.x = 0.2
    vel_cmd.angular.z = ang_vel
    self.vel_pub.publish(vel_cmd)

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
        # Straight reward = 5, Max angle reward = 0.5
        reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
        # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
    else:
        reward = -200

    return np.asarray(state), reward, done, {}


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

    return np.asarray(state)


if __name__ == '__main__':
    rospy.init_node('tcmdvel_publisher_dqn')

    vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    # env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    path = '/tmp/turtle_c2_dqn_ep'
    # plotter = liveplot.LivePlot(outdir)

    continue_execution = False

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 1000
        steps = 1000
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 100
        network_outputs = 21
        network_structure = [300,300]
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

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = reset()
        cumulated_reward = 0
        done = False
        episode_step = 0

        # run until env returns done
        while not done:
            # env.render()
            qValues = deepQ.getQValues(observation)

            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, newObservation, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)

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
                    # if (epoch)%100==0:
                    #     #save model weights and monitoring data every 100 epochs.
                    #     deepQ.saveModel(path+str(epoch)+'.h5')
                    #     env._flush()
                    #     copy_tree(outdir,path+str(epoch))
                    #     #save simulation parameters.
                    #     parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                    #     parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                    #     parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                    #     with open(path+str(epoch)+'.json', 'w') as outfile:
                    #         json.dump(parameter_dictionary, outfile)

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

            episode_step += 1

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

    #     if epoch % 100 == 0:
    #         plotter.plot(env)
    #
    # env.close()
