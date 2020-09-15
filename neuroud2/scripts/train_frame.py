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

import matplotlib.pyplot as plt

out_path = 'output.txt'
loss_out_path = 'output_loss.txt'
is_training = True

continue_execution = False
weight_file = "1100.h5"
params_json  = '1100.json'

def plot(x1,y1,x2,y2,cumulated_reward):
    xx.append(epoch+1)
    y.append(cumulated_reward)
    plt.plot(xx,y, color="blue")
    plt.draw()
    plt.pause(0.1)

if __name__ == '__main__':
    rospy.init_node('train_frame')

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    # env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
    outdir = '/tmp/gazebo_gym_experiments/'
    path = '/tmp/'
    # plotter = liveplot.LivePlot(outdir)

    env = Env(is_training)

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 10000
        steps = 300
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 541 + 1 + 2 + 1
        network_outputs = 7

        ### number of hiddenLayer ###
        network_structure = [300,21]
        current_epoch = 0

        deepQ = my_dqn.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        with open(params_json) as outfile:
            d = json.load(outfile)
            epochs = d.get('epochs')
            steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_inputs = d.get('network_inputs')
            network_outputs = d.get('network_outputs')
            network_structure = d.get('network_structure')
            current_epoch = d.get('current_epoch')

        deepQ = my_dqn.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
        print ("    ***** load file "+ weight_file+" *****")
        deepQ.loadWeights(weight_file)

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
    plt.xlim(0,2000)
    plt.ylim(-1000,2000)
    plt.grid()
    xx = []
    y = []
    y2 = []

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()
        cumulated_reward = 0
        done = False
        episode_step = 0
        last_action = 0
        service_count = 0

        # run until env returns done
        while not done:
            # env.render()
            qValues = deepQ.getQValues(observation)
            # print ("ss" ,qValues)

            action = deepQ.selectAction(qValues, explorationRate)

            # newObservation, reward, done, info = step(action, last_action)
            newObservation, reward, done, arrive, reach  = env.step(action, last_action)
            last_action = action

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, newObservation, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    history = deepQ.learnOnMiniBatch(minibatch_size, False)
                    # print "pass False"
                else :
                    history = deepQ.learnOnMiniBatch(minibatch_size, True)
                    # print "pass True"
                # print (str(history.history['loss']))
                filehandle = open(loss_out_path, 'a+')
                filehandle.write(str(epoch) + ',' + str(history.history['loss']) + "\n")
                filehandle.close()

            observation = newObservation

            episode_step += 1

            if reward == 200:
                service_count += 1

            if done or episode_step >= 300:
                done = True
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step) + "/" + str(steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))
                else :
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print ("EP " + str(epoch) + " - " + format(episode_step) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%50==0:
                        print ("save model")
                        deepQ.saveModel(str(epoch)+'.h5')
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(str(epoch)+'.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)


            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

        plot(xx,y,xx,y2,cumulated_reward)

        filehandle = open(out_path, 'a+')
        filehandle.write(str(epoch) + ',' + str(episode_step) + ',' + str(cumulated_reward)+ ',' + str(steps) +  ',' + str(service_count) + "\n")
        filehandle.close()


        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

    plt.savefig('output.png')
