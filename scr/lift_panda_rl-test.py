import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import sys
modelname = sys.argv[1]

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    control_freq=50,
    horizon = 200,
)
env = GymWrapper(env)
env = TimeFeatureWrapper(env)

model = SAC.load("./models/"+modelname, env = env)
done = False
n = 1
obs = env.reset()
while not done:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, _ = env.step(action)
    print("🔱", "{:03}".format(n), ["{:.4f}".format(x) for x in action], "{:.5f}".format(reward))
    env.unwrapped.render()
    n += 1
env.close()

# print("")
# print("🌼 [obs]")
# print("type(obs): ", type(obs))
# print("dir(obs): ", dir(obs))
# print("obs.shape: ", obs.shape)
# print("obs: ", obs)
# print("")

# print("🎃 robosuite.env")
# print("type(env): ", type(env))
# print("dir(env): ", dir(env))
# print("env: ", env)
# obs = env.reset()
# print("🌼 [obs] robosuite.env.reset()")
# print("type(obs): ", type(obs))
# print("dir(obs): ", dir(obs))
# for key,value in obs.items():
#     print(f"Key: {key}, Value.shape: {value.shape}")
# print("obs: ", obs)
# print("")

# env = GymWrapper(env)
# print("🎃 GymWrapper(env)")
# print("type(env): ", type(env))
# print("dir(env): ", dir(env))
# print("env: ", env)
# obs = env.reset()
# print("🌼 [obs] GymWrapper(env).reset()")
# print("type(obs): ", type(obs))
# print("dir(obs): ", dir(obs))
# print("obs.shape: ", obs.shape)
# print("obs: ", obs)
# print("")

# env = TimeFeatureWrapper(env)
# print("🎃 TimeFeatureWrapper(env)")
# print("type(env): ", type(env))
# print("dir(env): ", dir(env))
# print("env: ", env)
# obs = env.reset()
# print("🌼 [obs] TimeFeatureWrapper(env).reset()")
# print("type(obs): ", type(obs))
# print("dir(obs): ", dir(obs))
# print("obs.shape: ", obs.shape)
# print("obs: ", obs)
# print("")