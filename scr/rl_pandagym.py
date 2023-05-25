import gym
import panda_gym
import numpy as np
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3 .common.noise import NormalActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper

rb_kwargs = {'online_sampling' : True,
             'goal_selection_strategy' : 'future',
             'n_sampled_goal' : 4}

policy_kwargs = {'net_arch' : [512, 512, 512], 
                 'n_critics' : 2}

env = gym.make("PandaReach-v2")
env = TimeFeatureWrapper(env)

n_actions = env.action_space.shape[0]
noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1, 
             gamma = 0.95, batch_size= 2048, buffer_size=100000, replay_buffer_kwargs = rb_kwargs,
             learning_rate = 1e-3, action_noise = noise, policy_kwargs = policy_kwargs)
model.learn(1e3)
model.save('pick_place/model')
print("Saved!!")


model = DDPG.load("pick_place/model", env = env)
frames = []
for _ in range(10):
    done = False
    observation = env.reset()
    while not done:
        action, _states = model.predict(observation, deterministic = True)
        observation, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        frames.append(frame)
env.close()