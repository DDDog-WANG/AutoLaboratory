import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3 import DDPG , SAC, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from robosuite.utils.transform_utils import quat2mat, mat2euler
import argparse
import imageio
from tqdm import tqdm
import torch
np.set_printoptions(precision=5, suppress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str)
    parser.add_argument("--model_load", type=str)
    parser.add_argument("--model_name", type=str, default="SAC")
    parser.add_argument("--policy", type=str, default="small")

    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="rl_video")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()

controller_config = load_controller_config(default_controller=args.controller)
env_recoder = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    control_freq=50,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
    render_gpu_device_id=0,
    horizon=args.horizon,
    initialization_noise=None
)
env = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    render_gpu_device_id=0,
    horizon=args.horizon,
    initialization_noise=None
)
env = GymWrapper(env)
env = TimeFeatureWrapper(env)

writer = imageio.get_writer(args.workdir+"/videos_tmp/"+args.video_name+".mp4", fps=args.fps)

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
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


if args.model_name == "DDPG":
    model = DDPG(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs)
elif args.model_name == "SAC":
    model = SAC(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs)
elif args.model_name == "PPO":
    model = PPO(policy=CustomActorCriticPolicy, env=env)
model.policy.load_state_dict(torch.load(args.model_load))

obs = env.reset()
rewards = 0

for n in range(args.horizon):
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, _ = env.step(action)
    print("🔱", "{:03}".format(n), "{:.5f}".format(reward), flush=True)
    rewards += reward

    obs_recoder, reward_recorder, _, _ = env_recoder.step(action)
    # env.unwrapped.render()
    frame = obs_recoder[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
    if env_recoder._check_success(): break
print(env.robots[0].sim.data.qpos)

env.close()
env_recoder.close()
writer.close()
print(f"🔱 FINISH")
print(args.video_name)
print(f"rewards: {rewards}, steps: {n+1}, avg_rewards: {rewards/(n+1)}\n")
