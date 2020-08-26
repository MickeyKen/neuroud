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
state_dim = 1080 + 4
action_dim = 4
action_linear_max = 1.  # m/s
action_angular_max = 0.5  # rad/s
is_training = True
PAN_LIMIT = math.radians(90)  #2.9670
TILT_LIMIT = 1.3
TILT_MIN_LIMIT = math.radians(90) - math.atan(3.0/0.998)
TILT_MAX_LIMIT = math.radians(90) - math.atan(1.5/0.998)

EPISODES = 100000
steps = 500
TEST = 3

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
    past_action = np.array([0., 0., 0., 0.])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    var = 1.
    past_action = np.array([0., 0., 0., 0.])
    for episode in range(EPISODES):
        state = env.reset()
        pan_ang, tilt_ang = 0. ,0.

        for step in range(steps):
            a = agent.action(state)
            a[0] = np.clip(np.random.normal(a[0], var), 0., 1.)
            a[1] = np.clip(np.random.normal(a[1], var), -0.5, 0.5)
            a[2] = np.clip(np.random.normal(a[2], var), -PAN_LIMIT, PAN_LIMIT)
            a[3] = np.clip(np.random.normal(a[3], var), TILT_MIN_LIMIT, TILT_MAX_LIMIT)

            # a[2] = constrain(pan_ang + a[2], -PAN_LIMIT, PAN_LIMIT)
            # a[3] = constrain(tilt_ang + a[3], TILT_MIN_LIMIT, TILT_MAX_LIMIT)
            # pan_ang += a[2]
            # tilt_ang += a[3]

            next_state, r, done, arrive, reach = env.step(a, past_action)
            time_step = agent.perceive(state, a, r, next_state, done)
            state = next_state

            if done:
                break
        # Testing:
        if episode % 10 == 0 and episode > 10:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(steps):
                    a = agent.action(state)
                    a[0] = np.clip(np.random.normal(a[0], var), 0., 1.)
                    a[1] = np.clip(np.random.normal(a[1], var), -0.5, 0.5)
                    a[2] = np.clip(np.random.normal(a[2], var), -0.1, 0.1)
                    a[3] = np.clip(np.random.normal(a[3], var), -0.1, 0.1)

                    a[2] = constrain(past_action[2] + a[2], -PAN_LIMIT, PAN_LIMIT)
                    a[3] = constrain(past_action[3] + a[3], TILT_MIN_LIMIT, TILT_MAX_LIMIT)

                    state, r, done, arrive, reach = env.step(a, past_action)
                    total_reward += r
                    if done:
                        break
            ave_reward = total_reward/TEST
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)

if __name__ == '__main__':
    main()
