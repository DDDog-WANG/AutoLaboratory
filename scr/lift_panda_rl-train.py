import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
from stable_baselines3 import DDPG , SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import sys
workdir = sys.argv[1]
saveto = sys.argv[2]
horizon = 500
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=50,
    horizon=horizon,
)
env = GymWrapper(env)
env = TimeFeatureWrapper(env)

batch_size = 4096
learning_rate = 1e-3
episodes = 200
total_timesteps = horizon * episodes
policy_kwargs = {'net_arch' : [128, 256, 256, 256, 256, 128], 
                'n_critics' : 2,
                }
n_actions = env.robots[0].action_dim
print("n_actions: ", n_actions)
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2)
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=0.1, sigma=0.2)

# model = DDPG(policy="MlpPolicy", env=env, replay_buffer_class=ReplayBuffer, verbose=1, gamma = 0.95, batch_size=batch_size, 
#              buffer_size=100000, learning_rate = learning_rate, action_noise = action_noise, policy_kwargs = policy_kwargs)
model = SAC(policy="MlpPolicy", env=env, replay_buffer_class=ReplayBuffer, verbose=1, gamma = 0.95, batch_size=batch_size, 
            buffer_size=100000, learning_rate = learning_rate, action_noise = action_noise, policy_kwargs = policy_kwargs)
# model = SAC.load(workdir+'/models/SAC_big', env = env, learning_rate = learning_rate, action_noise = action_noise)
model.learn(total_timesteps=total_timesteps)
model.save(saveto)
print("Saved to ", saveto)
