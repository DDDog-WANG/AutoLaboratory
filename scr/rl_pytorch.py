from math import pi
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import mat2euler, quat2mat

controller_config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    env_name="TwoArmLift",
    robots="Maholo",
    gripper_types=["PandaGripper", "PandaGripper"],
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    control_freq=50,
    horizon = 500,
)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )
    def forward(self, state):
        action = self.layers(state) * self.action_bound
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        value = self.layers(x)
        return value

from collections import deque
from robosuite.wrappers import GymWrapper

def train(actor, critic, env, num_episodes=5000, buffer_size=100000, batch_size=64, gamma=0.99, tau=0.005):
    # 初始化环境
    env = GymWrapper(suite.make("TwoArmLift", robots="Baxter", has_renderer=False, has_offscreen_renderer=False))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # 创建目标网络
    target_actor = Actor(state_dim, action_dim, action_bound)
    target_critic = Critic(state_dim, action_dim)

    # 初始化经验回放缓冲区
    buffer = deque(maxlen=buffer_size)

    # 为优化器定义学习率
    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = actor(state_tensor).detach().numpy().squeeze()

            next_state, reward, done, _ = env.step(action)

            # 将经验添加到缓冲区
            buffer.append((state, action, reward, next_state, done))

            # 更新状态
            state = next_state
            episode_reward += reward

            # 如果缓冲区足够大，进行网络更新
            if len(buffer) > batch_size:
                # 从缓冲区中采样一个批次的经验
                batch = random.sample(buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.FloatTensor(actions)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

                # 更新Critic网络
                target_actions = target_actor(next_states_tensor).detach()
                target_q_values = target_critic(next_states_tensor, target_actions).detach()
                y = rewards_tensor + (1 - dones_tensor) * gamma * target_q_values

                q_values = critic(states_tensor, actions_tensor)
                critic_loss = nn.MSELoss()(q_values, y)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # 更新Actor网络
                actor_loss = -critic(states_tensor, actor(states_tensor)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 更新目标网络
                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if done:
                break

        print(f"Episode {episode}: Reward = {episode_reward}")

# 创建环境
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# 创建Actor和Critic网络
actor = Actor(state_dim, action_dim, action_bound)
critic = Critic(state_dim, action_dim)

# 开始训练
train(actor, critic, env)


