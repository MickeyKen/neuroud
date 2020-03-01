#!/usr/bin/env python
import gym
from gym import wrappers
# import gym_gazebo
import time
import numpy
import random
import time
# import liveplot
import qlearn
import myenv



if __name__ == '__main__':

    env = gym.make('GazeboUdLidar-v0')


    # outdir = '/tmp/gazebo_gym_experiments'
    # env = gym.wrappers.Monitor(env, outdir, force=True)


    last_time_steps = numpy.ndarray(0)

    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=0.1, gamma=0.8, epsilon=0.9)

    initial_epsilon = qlearn.epsilon

    epsilon_discount = 0.999 # 1098 eps to reach 0.1

    start_time = time.time()
    total_episodes = 10000
    highest_reward = 0

    for x in range(total_episodes):
        done = False

        cumulated_reward = 0 #Should going forward give more reward then L/R ?

        observation = env.reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        #render() #defined above, not env.render()

        state = ''.join(map(str, observation))

        for i in range(500):

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            qlearn.learn(state, action, reward, nextState)

            # env._flush(force=True)

            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

    env.close()
