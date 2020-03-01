#!/usr/bin/env python
import numpy as np
import rospy
import time
import random
import time
# import liveplot
import qlearn


def discretize_observation(self,data,new_ranges):
    discretized_ranges = []
    min_range = 0.2
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

def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

def step(self, action):

    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        self.unpause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/unpause_physics service call failed")

    if action == 0: #FORWARD
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.3
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
    elif action == 1: #LEFT
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.05
        vel_cmd.angular.z = 0.3
        self.vel_pub.publish(vel_cmd)
    elif action == 2: #RIGHT
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.05
        vel_cmd.angular.z = -0.3
        self.vel_pub.publish(vel_cmd)

    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        except:
            pass

    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        #resp_pause = pause.call()
        self.pause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/pause_physics service call failed")

    state,done = self.discretize_observation(data,5)

    if not done:
        if action == 0:
            reward = 5
        else:
            reward = 1
    else:
        reward = -200

    return state, reward, done, {}

def reset(self):

    # Resets the state of the environment and returns an initial observation.
    rospy.wait_for_service('/gazebo/reset_simulation')
    try:
        #reset_proxy.call()
        self.reset_proxy()
    except (rospy.ServiceException) as e:
        print ("/gazebo/reset_simulation service call failed")

    # Unpause simulation to make observation
    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        #resp_pause = pause.call()
        self.unpause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/unpause_physics service call failed")

    #read laser data
    data = None
    while data is None:
        try:
            data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        except:
            pass

    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        #resp_pause = pause.call()
        self.pause()
    except (rospy.ServiceException) as e:
        print ("/gazebo/pause_physics service call failed")

    state = self.discretize_observation(data,5)

    return state

if __name__ == '__main__':


    last_time_steps = numpy.ndarray(0)

    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=0.2, gamma=0.8, epsilon=0.9)

    initial_epsilon = qlearn.epsilon

    epsilon_discount = 0.9986

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

        for i in range(1500):

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            qlearn.learn(state, action, reward, nextState)

            env._flush(force=True)

            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        if x%100==0:
            plotter.plot(env)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    #Github table content
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
