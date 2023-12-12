from math import pi, degrees
import numpy as np
import time
import json
from tqdm import tqdm
import argparse
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
from robosuite.wrappers.gym_wrapper import GymWrapper
from sb3_contrib.common.wrappers import TimeFeatureWrapper
np.set_printoptions(precision=4, suppress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory_eefR_Move2Pipette")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()


controller_config = load_controller_config(default_controller=args.controller)
env = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    render_camera=args.camera,
    use_object_obs=True,
    use_camera_obs=False,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
    control_freq=50,
    horizon=args.horizon,
    initialization_noise=None,
)

for key in env.robots[0].gripper:
    print(f"{key} hand: {env.robots[0].gripper[key]}")


def print_joint_positions(joint_positions):
    print(f"👑 env.robots[0].sim.data.qpos.shape: {joint_positions.shape}")
    print("✤✤ Robot ✤✤")
    print("body         :", joint_positions[0])
    print("left_arm     :", joint_positions[1:8])
    print("right_arm    :", joint_positions[10:17])
    print("left_gripper :", joint_positions[8:10])
    print("right_gripper:", joint_positions[17:19])
    print("✤✤ Object ✤✤")
    print("pipette004_pos :", joint_positions[19:22])
    print("pipette004_quat:", joint_positions[22:26])
    print("tube008_pos    :", joint_positions[26:29])
    print("tube008_quat   :", joint_positions[29:33])

env.reset()
print_joint_positions(env.robots[0].sim.data.qpos)
for n in range(100):

    action = np.random.uniform(-1, 1, 17)

    obs, reward, done, _ = env.step(action)

    env.render()

env.close()