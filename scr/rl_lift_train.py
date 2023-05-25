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
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=50,
    horizon = 500,
)
env = GymWrapper(env)
env = TimeFeatureWrapper(env)


policy_kwargs = {'net_arch' : [512, 512, 512], 
                'n_critics' : 3}

n_actions = env.robots[0].action_dim
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG(policy="MlpPolicy", env=env, replay_buffer_class=ReplayBuffer, verbose=1,
            gamma = 0.95, batch_size= 2048, buffer_size=100000, learning_rate = 1e-5, action_noise = action_noise, policy_kwargs = policy_kwargs)
model.learn(total_timesteps=1000000)
model.save(workdir+'/models/DDPG')
print("Saved to ", workdir+'/models/DDPG')
