import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
import inspect
import torch
from torch import nn
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3 import DDPG , SAC, PPO, HerReplayBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.evaluation import evaluate_policy
import argparse, os
import datetime, time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default="./")
    parser.add_argument("--model_save", type=str, default="my_model")
    parser.add_argument("--model_load", type=str, default=None)
    parser.add_argument("--log_save", type=str, default="./")

    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)

    parser.add_argument("--model_name", type=str, default="SAC")
    parser.add_argument("--policy", type=str, default="small")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--final_lr", type=float, default=1e-5)
    parser.add_argument("--initial_sigma", type=float, default=0.2)
    parser.add_argument("--final_sigma", type=float, default=0.02)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    args = parser.parse_args()

controller_config = load_controller_config(default_controller=args.controller)
env = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_object_obs=True,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    render_gpu_device_id=0,
    horizon=args.horizon,
    initialization_noise=None
)
for key,value in env.reset().items():
    print(f"Key: {key}, Value.shape: {value.shape}", flush=True)
env = GymWrapper(env)
print(f"\nGYM Wrapper obs: {env.reset().shape}\n", flush=True)
# env = TimeFeatureWrapper(env)
# print(f"\nTimeFeature GYM Wrapper obs: {env.reset().shape}\n", flush=True)
env_test = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_object_obs=True,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    render_gpu_device_id=0,
    horizon=args.horizon,
    initialization_noise=None
)
env_test = GymWrapper(env_test)
# CONTROLLER
if args.controller == "OSC_POSE":
    n_actions = 14
else:
    n_actions = env.robots[0].action_dim
print(f"n_actions: {n_actions}\n", flush=True)

# POLICY NETWORK
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.block(x)
class CustomNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
        
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi),
            nn.ReLU(),
            ResidualBlock(last_layer_dim_pi),
            nn.ReLU(),
            ResidualBlock(last_layer_dim_pi),
            nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf),
            nn.ReLU(),
            ResidualBlock(last_layer_dim_vf),
            nn.ReLU(),
            ResidualBlock(last_layer_dim_pi),
            nn.ReLU(),
        )
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
    def _build_mlp_extractor(self):
        self.mlp_extractor = CustomNetwork(self.features_dim)
if args.policy == "large":
    policy_kwargs = {'net_arch' : [512, 512, 512, 512, 256, 256, 128, 128], 
                    'n_critics' : 4,
                    }
elif args.policy == "middle":
    policy_kwargs = {'net_arch' : [512, 512, 512, 512], 
                    'n_critics' : 4,
                    }
elif args.policy == "small":
    policy_kwargs = {'net_arch' : [512, 512], 
                    'n_critics' : 2,
                    }

# ACTION NOISE
sigma_schedule = lambda fraction: args.initial_sigma + fraction * (args.final_sigma - args.initial_sigma)
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.initial_sigma)

# LEARNING RATE
save_interval = args.save_interval
total_timesteps_per_iter = args.horizon * save_interval
total_timesteps = 4 * args.horizon * args.episodes
# decay_rate = -np.log(args.final_lr / args.initial_lr) / total_timesteps
# lr_schedule = lambda step: args.initial_lr * np.exp(decay_rate * step)
lr_schedule = lambda fraction: args.initial_lr + (1-fraction) * (args.final_lr - args.initial_lr)

# ALGORITHM
if args.model_name == "DDPG":
    model = DDPG(policy="MlpPolicy", policy_kwargs=policy_kwargs, env=env, verbose=1, gamma=0.9, batch_size=args.batch_size, action_noise=action_noise, 
                 replay_buffer_class=ReplayBuffer, learning_rate=lr_schedule, tensorboard_log=args.log_save, device="cuda")
elif args.model_name == "SAC":
    model = SAC(policy="MlpPolicy", policy_kwargs=policy_kwargs, env=env, verbose=1, gamma = 0.9, batch_size=args.batch_size, action_noise=action_noise, 
                replay_buffer_class=ReplayBuffer, learning_rate=lr_schedule, tensorboard_log=args.log_save, device="cuda")
elif args.model_name == "PPO":
    model = PPO(policy=CustomActorCriticPolicy, env=env, learning_rate=lr_schedule, verbose=1, gamma=0.9, batch_size=args.batch_size, 
                tensorboard_log=args.log_save, device="cuda")
if args.model_load is not None:
    if os.path.exists(args.model_load):
        try:
            model.policy.load_state_dict(torch.load(args.model_load))
            print(f"Model weights loaded from {args.model_load}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"Model weights file {args.model_load} does not exist.")

print("\n🔱 MODEL POLICY")
print(model.policy, flush=True)
print("\n💰 Reward Function")
print(inspect.getsource(env.unwrapped.reward), flush=True)
best_reward = -np.inf
for i in range(args.episodes // save_interval):
    print("✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢")
    print("✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤")
    start_time = time.time()
    print(f"👑 ROUND {i * save_interval} ", datetime.datetime.now(), flush=True)

    # Train model
    model.action_noise._sigma = sigma_schedule(i/(args.episodes // save_interval))
    print(f"model.action_noise._sigma: {model.action_noise._sigma}", flush=True)
    model.learn(total_timesteps=total_timesteps_per_iter, reset_num_timesteps=False)

    # Save model
    rewards = 0
    obs = env_test.reset()
    for n in range(args.horizon):
        action, _states = model.predict(obs, deterministic = True)
        obs, reward, done, _ = env_test.step(action)
        rewards += reward
        if env_test._check_success():
            print(f"🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
            save_path = args.model_save+f'_succeed_{i*save_interval}.pth'
            torch.save(model.policy.state_dict(), save_path)
            print(f"Succeed in {n} Steps, Saved to {save_path}\n", flush=True)
            break
    
    if rewards > best_reward:
        save_path = args.model_save+'.pth'
        torch.save(model.policy.state_dict(), save_path)
        print(f"🤩 New best rewards is {rewards}, better than last reward {best_reward}, Saved to {save_path}", flush=True)
        best_reward = rewards
    else: 
        print(f"😰 Rewards this time is {rewards}, not better than last reward: {best_reward}", flush=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, _ = divmod(remainder, 60)
    print(f"Time elapsed: {int(hours)} hours and {int(minutes)} minutes")
    print(f"ROUND {i * save_interval} FINISH", datetime.datetime.now(), flush=True)

# model.learn(total_timesteps=total_timesteps)
# torch.save(model.policy.state_dict(), args.model_save)
# print("Saved to ", args.model_save, flush=True)




