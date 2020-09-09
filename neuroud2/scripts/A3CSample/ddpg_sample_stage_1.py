#!/usr/bin/env python
import rospy
import gym
import math
# import gym_gazebo
import numpy as np
import tensorflow as tf
from ddpg import *
from environment import Env


exploration_decay_start_step = 50000
state_dim = 541 + 3
action_dim = 3
action_linear_max = 1.  # m/s
action_angular_max = 0.5  # rad/s
is_training = True
PAN_LIMIT = math.radians(90)  #2.9670
TILT_LIMIT = 1.3
TILT_MIN_LIMIT = math.radians(90) - math.atan(3.0/0.998)
TILT_MAX_LIMIT = math.radians(90) - math.atan(1.5/0.998)

EPISODES = 10000
TEST = 3

is_training = True

path = 'output.txt'

def constrain(input, low, high):
    if input < low:
      input = low
    elif input > high:
      input = high
    else:
      input = input

    return input

def main():
    rospy.init_node('ddpg_sample_stage_1')
    env = Env(is_training)
    agent = DDPG(env, state_dim, action_dim)
    past_action = np.array([0., 0., 0.])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    var = 1.

    if is_training:
        for episode in range(EPISODES):
            state = env.reset()
            past_action = np.array([0., 0., 0.])
            one_round_step = 0
            cumulated_reward = 0.
            reach_count = 0

            while True:
                a = agent.action(state)
                a[0] = np.clip(np.random.normal(a[0], var), -1., 1.)
                a[1] = np.clip(np.random.normal(a[1], var), -PAN_LIMIT, PAN_LIMIT)
                a[2] = np.clip(np.random.normal(a[2], var), TILT_MIN_LIMIT, TILT_MAX_LIMIT)

                state_, r, done, arrive, reach = env.step(a, past_action)
                time_step = agent.perceive(state, a, r, state_, done)
                state = state_
                past_action = a

                cumulated_reward += r

                if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    var *= 0.9999

                one_round_step += 1

                if reach == True and arrive == True:
                    reach_count += 1

                if done or one_round_step >= 300:
                    print ('Finished episode: ',episode,'Compulated Reward: ',cumulated_reward, 'Serviced count: ', reach_count)
                    print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', "Fail")
                    filehandle = open(path, 'a+')
                    filehandle.write(str(episode) + ',' + str(one_round_step) + ',' + str(cumulated_reward)+ ',' + str(reach_count) + "\n")
                    filehandle.close()
                    break

            # Testing:
            # if episode % 10 == 0 and episode >= 10:
            #     total_reward = 0
            #     for i in range(TEST):
            #         state = env.reset()
            #         past_action = np.array([0., 0., 0.])
            #         one_round_step = 0
            #
            #         for j in range(steps):
            #             a = agent.action(state)
            #             a[0] = np.clip(np.random.normal(a[0], var), -1., 1.)
            #             a[1] = np.clip(np.random.normal(a[1], var), -0.15, 0.15)
            #             a[2] = np.clip(np.random.normal(a[2], var), -0.15, 0.15)
            #             state_, r, done, arrive, reach = env.step(a, past_action)
            #             total_reward += r
            #             past_action = a
            #             state = state_
            #             one_round_step += 1
            #
            #             if done or one_round_step >= 500:
            #                 print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', "Fail")
            #                 break
            #
            #     ave_reward = total_reward/TEST
            #     print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
                #
                # filehandle = open(path, 'a')
                # filehandle.write(str(episode) + ',' + str(ave_reward)+"\n")
                # filehandle.close()
            #
            # else:
            #     print ('Finished episode: ',episode)

    else:
        print('Testing mode')
        for episode in range(EPISODES):
            state = env.reset()
            past_action = np.array([0., 0., 0.])
            one_round_step = 0
            total_reward = 0

            while True:
                a = agent.action(state)
                a[0] = np.clip(np.random.normal(a[0], var), -1., 1.)
                a[1] = np.clip(np.random.normal(a[1], var), -0.15, 0.15)
                a[2] = np.clip(np.random.normal(a[2], var), -0.15, 0.15)
                state_, r, done, arrive, reach = env.step(a, past_action)
                total_reward += r
                past_action = a
                state = state_
                one_round_step += 1

                if done or one_round_step >= 500:
                    print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', "Fail")
                    break

            ave_reward = total_reward/TEST
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)

            filehandle = open(path, 'a')
            filehandle.write('episode: '+str(episode)+'  Evaluation Average Reward:'+str(ave_reward)+"\n")
            filehandle.close()
if __name__ == '__main__':
    main()
