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

from train_environment_1 import Env1
from train_environment_2 import Env2

import matplotlib.pyplot as plt

out_path = 'output.txt'
loss_out_path = 'output_loss.txt'
is_training = True

continue_execution = False
weight_file = "450.h5"
params_json  = '450.json'

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

    env1 = Env1(is_training)
    env2 = Env2(is_training)

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 3000000
        steps = 300
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 541 + 1 + 2 + 1
        network_outputs = 8

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
    #
    # plt.ion()
    # plt.title('Simple Curve Graph')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.xlim(0,2000)
    # plt.ylim(-1000,2000)
    # plt.grid()
    xx = []
    y = []
    y2 = []

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation1 = env1.reset()
        observation2 = env2.reset()
        cumulated_reward1 = 0
        cumulated_reward2 = 0
        done1 = False
        done2 = False
        episode_step = 0
        last_action1 = 0
        last_action2 = 0
        service_count = 0
        loss_sum = 0.0

        # run until env returns done
        while not done1 or not done2:
            # env.render()

            if not done1:
                qValues1 = deepQ.getQValues(observation1)
                action1 = deepQ.selectAction(qValues1, explorationRate)
                newObservation1, reward1, done1, arrive1, reach1  = env1.step(action1, last_action1)
                last_action1 = action1
                cumulated_reward1 += reward1
                if highest_reward < cumulated_reward1:
                    highest_reward = cumulated_reward1
                deepQ.addMemory(observation1, action1, reward1, newObservation1, done1)
                observation1 = newObservation1

            if not done2:
                qValues2 = deepQ.getQValues(observation2)
                action2 = deepQ.selectAction(qValues2, explorationRate)
                newObservation2, reward2, done2, arrive2, reach2  = env2.step(action2, last_action2)
                last_action2 = action2
                cumulated_reward2 += reward2
                if highest_reward < cumulated_reward2:
                    highest_reward = cumulated_reward2
                deepQ.addMemory(observation2, action2, reward2, newObservation2, done2)
                observation2 = newObservation2

            # qValues1 = deepQ.getQValues(observation1)
            # qValues2 = deepQ.getQValues(observation2)
            #
            # action1 = deepQ.selectAction(qValues1, explorationRate)
            # action2 = deepQ.selectAction(qValues2, explorationRate)
            #
            # newObservation1, reward1, done1, arrive1, reach1  = env1.step(action1, last_action1)
            # newObservation2, reward2, done2, arrive2, reach2  = env2.step(action2, last_action2)
            #
            # last_action1 = action1
            # last_action2 = action2
            #
            # cumulated_reward1 += reward1
            # if highest_reward < cumulated_reward1:
            #     highest_reward = cumulated_reward1
            # cumulated_reward2 += reward2
            # if highest_reward < cumulated_reward2:
            #     highest_reward = cumulated_reward2
            #
            # deepQ.addMemory(observation1, action1, reward1, newObservation1, done1)
            # deepQ.addMemory(observation2, action2, reward2, newObservation2, done2)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    history = deepQ.learnOnMiniBatch(minibatch_size, False)
                    # print "pass False"
                else :
                    history = deepQ.learnOnMiniBatch(minibatch_size, True)
                    # print "pass True"
                loss_sum += history.history['loss'][0]
                # print (str(history.history['loss']))

            # observation1 = newObservation1
            # observation2 = newObservation2

            episode_step += 1

            if reward1 == 200 or reward2 == 200:
                service_count += 1

            if (done1 and done2) or episode_step >= 300:
                done1 = True
                done2 = True
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
                    print ("EP " + str(epoch) + " - " + format(episode_step) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated1 R: " + str(cumulated_reward1) + " - Cumulated2 R: " + str(cumulated_reward2) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%50==0:
                        print ("save model")
                        deepQ.saveModel(str(epoch)+'.h5')
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(str(epoch)+'.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)


            stepCounter += 2
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

        # plot(xx,y,xx,y2,cumulated_reward)

        filehandle = open(out_path, 'a+')
        filehandle.write(str(epoch) + ',' + str(episode_step) + ',' + str(cumulated_reward1)+ ',' + str(cumulated_reward2)+ ',' + str(steps) +  ',' + str(service_count) + "\n")
        filehandle.close()
        filehandle = open(loss_out_path, 'a+')
        filehandle.write(str(epoch) + ',' + str(loss_sum/float(episode_step)) + "\n")
        filehandle.close()


        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

    plt.savefig('output.png')
