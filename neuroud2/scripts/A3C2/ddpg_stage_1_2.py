#!/usr/bin/env python
import rospy
import gym
# import gym_gazebo
import numpy as np
import tensorflow as tf
from ddpg2 import *
from environment2 import Env
import math

exploration_decay_start_step = 50000
state_dim = 1083 + 4
action_dim = 2
is_training = True

action_pan_max = 0.15
action_tilt_max = 0.15

PAN_LIMIT = math.radians(90)  #2.9670
TILT_MIN_LIMIT = math.radians(90) - math.atan(3.0/0.998)
TILT_MAX_LIMIT = math.radians(90) - math.atan(1.5/0.998)

path = 'output.txt'

def main():
    rospy.init_node('ddpg_stage_1')
    env = Env(is_training)
    agent = DDPG(env, state_dim, action_dim)
    past_action = np.array([0., TILT_MIN_LIMIT])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: (Pan)' + str(action_pan_max) + ' rad/s and (Tilt)' + str(action_tilt_max) + ' rad/s')

    if is_training:
        print('Training mode')
        avg_reward_his = []
        total_reward = 0
        var = 1.

        episode = 0
        while True:
            state = env.reset()
            one_round_step = 0
            past_action = np.array([0., TILT_MIN_LIMIT])
            cumulated_reward = 0.
            reach_count = 0

            while True:
                a = agent.action(state)
                a[0] = np.clip(np.random.normal(a[0], var), -0.15, 0.15)
                a[1] = np.clip(np.random.normal(a[1], var), -0.15, 0.15)

                state_, r, reach  = env.step(a, past_action)
                time_step = agent.perceive(state, a, r, state_, reach)

                cumulated_reward += r
                if reach:
                    reach_count += 1

                if time_step > 0:
                    total_reward += r

                if time_step % 10000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    avg_reward = total_reward / 10000
                    print('Average_reward = ', avg_reward)
                    avg_reward_his.append(round(avg_reward, 2))
                    print('Average Reward:',avg_reward_his)
                    total_reward = 0

                if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    var *= 0.9999

                past_action = a
                state = state_
                one_round_step += 1

                if one_round_step >= 500:
                    print ('Finished episode: ',episode,'Compulated Reward: ',cumulated_reward, 'Serviced count: ', reach_count)
                    print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', "Fail")
                    filehandle = open(path, 'a+')
                    filehandle.write(str(episode) + ',' + str(one_round_step) + ',' + str(cumulated_reward)+ ',' + str(reach_count) + "\n")
                    filehandle.close()
                    episode += 1
                    break

    else:
        print('Testing mode')
        while True:
            state = env.reset()
            one_round_step = 0

            while True:
                a = agent.action(state)
                a[0] = np.clip(np.random.normal(a[0], var), -1., 1.)
                a[1] = np.clip(np.random.normal(a[1], var), -1., 1.)
                a[2] = np.clip(np.random.normal(a[2], var), -2.9670, 2.9670)
                a[3] = np.clip(np.random.normal(a[3], var), -0.2617, 1.3)
                state_, r, done, arrive = env.step(a, past_action)
                past_action = a
                state = state_
                one_round_step += 1


                if arrive:
                    print('Step: %3i' % one_round_step, '| Arrive!!!')
                    one_round_step = 0

                if done:
                    print('Step: %3i' % one_round_step, '| Collision!!!')
                    break


if __name__ == '__main__':
     main()
