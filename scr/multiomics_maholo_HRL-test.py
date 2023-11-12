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
    parser.add_argument("--model_load_eefR_Move2Pipette", type=str)
    parser.add_argument("--model_load_eefR_Grip2Pipette", type=str)
    parser.add_argument("--model_name", type=str, default="SAC")
    parser.add_argument("--policy", type=str, default="middle")

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
    "MaholoLaboratory_eefR_Move2Pipette",
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
env_eefR_Move2Pipette = suite.make(
    "MaholoLaboratory_eefR_Move2Pipette",
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
env_eefR_Move2Pipette = TimeFeatureWrapper(GymWrapper(env_eefR_Move2Pipette))
env_eefR_Grip2Pipette = suite.make(
    "MaholoLaboratory_eefR_Grip2Pipette",
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
env_eefR_Grip2Pipette = TimeFeatureWrapper(GymWrapper(env_eefR_Grip2Pipette))
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

if args.model_name == "SAC":
    model_eefR_Move2Pipette = SAC(policy="MlpPolicy", env=env_eefR_Move2Pipette, policy_kwargs=policy_kwargs)
    model_eefR_Move2Pipette.policy.load_state_dict(torch.load(args.model_load_eefR_Move2Pipette))

    model_eefR_Grip2Pipette = SAC(policy="MlpPolicy", env=env_eefR_Grip2Pipette, policy_kwargs=policy_kwargs)
    model_eefR_Grip2Pipette.policy.load_state_dict(torch.load(args.model_load_eefR_Grip2Pipette))



obs = env_eefR_Move2Pipette.reset()
rewards = 0
n = 1
print("Agent MaholoLaboratory_eefR_Move2Pipette start working")
while not env_eefR_Move2Pipette.unwrapped._check_success() and n <= args.horizon:
    action, _states = model_eefR_Move2Pipette.predict(obs, deterministic = True)
    obs, reward, done, _ = env_eefR_Move2Pipette.step(action)    
    _, _, _, _           = env_eefR_Grip2Pipette.step(action)
    print("🔱", "{:03}".format(n), "Agent eefR_Move2Pipette", "{:.5f}".format(reward), flush=True)
    rewards += reward
    obs_recoder, reward_recorder, _, _ = env_recoder.step(action)
    frame = obs_recoder[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
    n += 1
print("Agent MaholoLaboratory_eefR_Grip2Pipette start working")
while not env_eefR_Grip2Pipette.unwrapped._check_success() and n <= args.horizon:
    action, _states = model_eefR_Grip2Pipette.predict(obs, deterministic = True)
    _, _, _, _           = env_eefR_Move2Pipette.step(action)    
    obs, reward, done, _ = env_eefR_Grip2Pipette.step(action)
    print("🔱", "{:03}".format(n), "Agent eefR_Grip2Pipette", "{:.5f}".format(reward), flush=True)
    rewards += reward
    obs_recoder, reward_recorder, _, _ = env_recoder.step(action)
    frame = obs_recoder[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
    n += 1
    
print(f"🔱 FINISH")
print(f"rewards: {rewards}, steps: {n-1}, avg_rewards: {rewards/(n-1)}\n")
env_eefR_Move2Pipette.close()
env_eefR_Grip2Pipette.close()
env_recoder.close()
writer.close()

