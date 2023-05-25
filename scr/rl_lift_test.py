import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import sys
workdir = sys.argv[1]

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    horizon = 500,
)
env = GymWrapper(env)
env = TimeFeatureWrapper(env)

model = DDPG.load(workdir+'/models/DDPG', env = env)
done = False
obs = env.reset()
while not done:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, _ = env.step(action)
    env.unwrapped.render()
env.close()