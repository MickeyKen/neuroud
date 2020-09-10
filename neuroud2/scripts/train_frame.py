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

from train_environment import Env

out_path = 'output.txt'
is_training = True

if __name__ == '__main__':
    rospy.init_node('train_frame')

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    # env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    path = '/tmp/'
    # plotter = liveplot.LivePlot(outdir)

    continue_execution = False
    env = Env(is_training)

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
        network_inputs = 541 + 1 + 2
        network_outputs = 7

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


    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()
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

            # newObservation, reward, done, info = step(action, last_action)
            newObservation, reward, done  = env.step(action, last_action)
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
                    filehandle = open(path, 'a+')
                    filehandle.write(str(epoch) + ',' + str(episode_step + 1) + ',' + str(cumulated_reward)+ ',' + str(steps) + "\n")
                    filehandle.close()
                    if (epoch)%100==0:
                        deepQ.saveModel(path+str(epoch)+'.h5')

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

            episode_step += 1


        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)
