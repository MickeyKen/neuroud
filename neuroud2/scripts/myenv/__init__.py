from gym.envs.registration import register

register(
    id='GazeboUdLidar-v0',
    entry_point='myenv.env:GazeboEnv',
    # More arguments here
)
