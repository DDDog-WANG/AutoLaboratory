import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np

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
)
env_eefR_Grip2Pipette = TimeFeatureWrapper(GymWrapper(env_eefR_Grip2Pipette))

batch_size = args.batch_size
# 计算衰减率
decay_rate = -np.log(args.final_lr / args.initial_lr)
lr_schedule = lambda fraction: args.initial_lr * np.exp(-decay_rate * (1-fraction))

total_timesteps = args.horizon * args.episodes
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
    
if args.controller == "OSC_POSE":
    n_actions = 14
else:
    n_actions = env_eefR_Move2Pipette.robots[0].action_dim
print(f"n_actions: {n_actions}\n", flush=True)
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2)


if args.model_name == "SAC":
    model_eefR_Move2Pipette = SAC(policy="MlpPolicy", env=env_eefR_Move2Pipette, policy_kwargs=policy_kwargs)
    model_eefR_Move2Pipette.policy.load_state_dict(torch.load(args.model_load_eefR_Move2Pipette))

    model_eefR_Grip2Pipette = SAC(policy="MlpPolicy", policy_kwargs=policy_kwargs, env=env, verbose=1, gamma = 0.9, batch_size=batch_size, action_noise=action_noise, 
            replay_buffer_class=ReplayBuffer, learning_rate=lr_schedule, tensorboard_log=args.log_save)
    model_eefR_Grip2Pipette.policy.load_state_dict(torch.load(args.model_load_eefR_Grip2Pipette))
else: print("model name is not SAC now")

print("\nMODEL POLICY")
print(model_eefR_Grip2Pipette.policy, flush=True)
save_interval = 1000
for i in range(args.episodes // save_interval):
    print("✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢")
    print("✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤")
    start_time = time.time()
    print(f"👑 ROUND {i * save_interval} ", datetime.datetime.now(), flush=True)

    obs = env_eefR_Move2Pipette.reset()
    for n in range(args.horizon):
        while not env_eefR_Move2Pipette.unwrapped._check_success():
            action, _states = model_eefR_Move2Pipette.predict(obs, deterministic = True)
            obs, reward, done, _ = env_eefR_Move2Pipette.step(action)    
            _, _, _, _           = env_eefR_Grip2Pipette.step(action)
            
    # Train model
    total_timesteps_per_iter = args.horizon * save_interval
    model_eefR_Grip2Pipette.learn(total_timesteps=total_timesteps_per_iter, reset_num_timesteps=False)

    # Save model
    save_path = args.model_save+f'_{(i + 1)*save_interval}.pth'
    torch.save(model_eefR_Grip2Pipette.policy.state_dict(), save_path)
    print(f"Saved to {save_path}\n", flush=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, _ = divmod(remainder, 60)
    print(f"Time elapsed: {int(hours)} hours and {int(minutes)} minutes")
    print(f"ROUND {i * save_interval} FINISH", datetime.datetime.now(), flush=True)

env_eefR_Move2Pipette.close()
env_eefR_Grip2Pipette.close()

