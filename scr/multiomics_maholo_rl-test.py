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
# np.set_printoptions(precision=5, suppress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str)
    parser.add_argument("--model_load", type=str)
    parser.add_argument("--model", type=str, default="SAC")
    parser.add_argument("--policy", type=str, default="small")
    parser.add_argument("--reward_version", type=str, default="0")

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
    use_object_obs=True,
    use_camera_obs=True,
    control_freq=50,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
    render_gpu_device_id=0,
    horizon=args.horizon,
    initialization_noise=None,
    reward_version=args.reward_version,
)
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
    initialization_noise=None,
    reward_version=args.reward_version,
)
env = GymWrapper(env)
# env = TimeFeatureWrapper(env)
writer = imageio.get_writer(f"{args.workdir}/videos_tmp/{args.video_name}.mp4", fps=args.fps)

# POLICY NETWORK
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

# ALGORITHM
if args.model == "DDPG":
    model = DDPG(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs)
elif args.model == "SAC":
    model = SAC(policy="MlpPolicy", env=env, policy_kwargs=policy_kwargs)
model.policy.load_state_dict(torch.load(args.model_load))

trajectory_joint_qpos = []
trajectory_actions = []
obs = env.reset()
def print_joint_positions(joint_positions):
    print(f"👑 env.robots[0].sim.data.qpos.shape: {joint_positions.shape}")
    print("body         :", joint_positions[0])
    print("left_arm     :", joint_positions[1:8])
    print("right_arm    :", joint_positions[10:17])
    print("left_gripper :", joint_positions[8:10])
    print("right_gripper:", joint_positions[17:19])
    print("pipette004_pos :", joint_positions[19:22])
    print("pipette004_quat:", joint_positions[22:26])
    print("tube008_pos    :", joint_positions[26:29])
    print("tube008_quat   :", joint_positions[29:33])
print_joint_positions(env.robots[0].sim.data.qpos)
print("📷 obs: ")
print(obs)
rewards = 0
for n in range(args.horizon):
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, _ = env.step(action)
    rewards += reward

    obs_recoder, reward_recorder, _, _ = env_recoder.step(action)
    print("🔱", "{:03}".format(n), "{:.5f}".format(reward), np.linalg.norm(obs_recoder["g1_to_target_pos"]), obs_recoder["g1_to_target_quat"], np.linalg.norm(obs_recoder["g0_to_target_pos"]), obs_recoder["g0_to_target_quat"], flush=True)

    frame = obs_recoder[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)

    trajectory_joint_qpos.append(env.robots[0].sim.data.qpos[:19].tolist())
    trajectory_actions.append(action)

    if env_recoder._check_success(): break

print_joint_positions(env.robots[0].sim.data.qpos)
print("📷 obs: ")
print(obs)

env.close()
env_recoder.close()
writer.close()
print(f"🔱 FINISH")
print(args.video_name)
print(f"rewards: {rewards}, steps: {n+1}, avg_rewards: {rewards/(n+1)}\n")

trajectory_joint_qpos = np.array(trajectory_joint_qpos)
print("trajectory_joint_qpos.shape: ", trajectory_joint_qpos.shape)
trajectory_actions = np.array(trajectory_actions)
print("trajectory_actions.shape: ", trajectory_actions.shape)
np.save(f"{args.workdir}/data/trajectory_joint_qpos-{args.video_name}.npy", trajectory_joint_qpos)
np.savetxt(f"{args.workdir}/data/trajectory_joint_qpos-{args.video_name}.txt", trajectory_joint_qpos)
np.save(f"{args.workdir}/data/trajectory_actions-{args.video_name}.npy", trajectory_actions)
np.savetxt(f"{args.workdir}/data/trajectory_actions-{args.video_name}.txt", trajectory_actions)

