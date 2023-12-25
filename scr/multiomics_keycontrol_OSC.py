from math import pi, degrees
import argparse
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import json, time, math
from copy import deepcopy
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler, quat_distance
from pynput import keyboard
from robosuite.wrappers.gym_wrapper import GymWrapper
from sb3_contrib.common.wrappers import TimeFeatureWrapper
# np.set_printoptions(precision=5, suppress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()

controller_config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
    horizon=args.horizon,
    initialization_noise=None
)
# env = GymWrapper(env) 
# env = TimeFeatureWrapper(env)

# sidecamera: dawsqe ikjlou10
delta = 0.2
arm_delta = 7
def on_press(key):
    global action
    try:
        if key.char == "w":
            action[0+arm_delta] = -delta
        elif key.char == "s":
            action[0+arm_delta] = delta
        elif key.char == "a":
            action[1+arm_delta] = -delta
        elif key.char == "d":
            action[1+arm_delta] = delta
        elif key.char == "q":
            action[2+arm_delta] = delta
        elif key.char == "e":
            action[2+arm_delta] = -delta

        elif key.char == "j":
            action[3+arm_delta] = delta/2
        elif key.char == "l":
            action[3+arm_delta] = -delta/2
        elif key.char == "k":
            action[4+arm_delta] = delta/2
        elif key.char == "i":
            action[4+arm_delta] = -delta/2
        elif key.char == "o":
            action[5+arm_delta] = delta/2
        elif key.char == "u":
            action[5+arm_delta] = -delta/2
        elif key.char == "1":
            action[6+arm_delta] = delta
        elif key.char == "0":
            action[6+arm_delta] = -delta

        elif key.char == "W":
            action[0] = -delta
        elif key.char == "S":
            action[0] = delta
        elif key.char == "A":
            action[1] = -delta
        elif key.char == "D":
            action[1] = delta
        elif key.char == "Q":
            action[2] = delta
        elif key.char == "E":
            action[2] = -delta

        elif key.char == "J":
            action[3] = delta/2
        elif key.char == "L":
            action[3] = -delta/2
        elif key.char == "K":
            action[4] = delta/2
        elif key.char == "I":
            action[4] = -delta/2
        elif key.char == "O":
            action[5] = delta/2
        elif key.char == "U":
            action[5] = -delta/2
        elif key.char == "!":
            action[6] = delta
        elif key.char == "~":
            action[6] = -delta
    except AttributeError:
        pass
def on_release(key):
    global action
    try:
        action[:] = 0
    except AttributeError:
        pass
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


# joint_names = env.sim.model.joint_names
# joint_ids = [env.sim.model.joint_name2id(name) for name in joint_names] 

action = np.zeros(14)
action_seq = []
action_seq_joint = []
obs_seq = []
reward_seq = []

def print_joint_positions(joint_positions):
    print("  robot_initial_qpos=np.array([", ', '.join(f"{x:.8f}" for x in joint_positions[0:8]), ", ", ', '.join(f"{x:.8f}" for x in joint_positions[10:17]), "]),")
    print("  gripper_l_initial_qpos = np.array([", ', '.join(f"{x:.8f}" for x in joint_positions[8:10]), "]),")
    print("  gripper_r_initial_qpos = np.array([", ', '.join(f"{x:.8f}" for x in joint_positions[17:19]), "]),")
    print("  tube008_initial_pos = np.array([", ', '.join(f"{x:.8f}" for x in joint_positions[19:22]), "]),")
    print("  tube008_initial_quat = np.array([", ', '.join(f"{x:.8f}" for x in joint_positions[22:26]), "]),")
    print("  pipette004_initial_pos = np.array([", ', '.join(f"{x:.8f}" for x in joint_positions[26:29]), "]),")
    print("  pipette004_initial_quat = np.array([", ', '.join(f"{x:.8f}" for x in joint_positions[29:33]), "]),")


obs = env.reset()
for key,value in obs.items():
    print(f"Key: {key}, Value.shape: {value.shape}")
    
print_joint_positions(env.robots[0].sim.data.qpos)
eef_pos = env.sim.data.get_body_xpos("gripper0_left_eef")
eef_euler = env.sim.data.get_body_xquat("gripper0_left_eef")
eef_pos = env.sim.data.get_body_xpos("gripper0_right_eef")
eef_euler = env.sim.data.get_body_xquat("gripper0_right_eef")
print(f"left_eef : {eef_pos}, {eef_euler}")
print(f"right_eef: {eef_pos}, {eef_euler}")
print("✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢")
print("✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✢✢✤✤")

for n in range(args.horizon):

    obs_seq.append(obs)
    action_seq.append(action.copy())

    obs, reward, done, _ = env.step(action)
    reward_seq.append(reward)
    print("🔱", "{:03}".format(n), "{:.5f}".format(reward), flush=True)
    # print(obs["pipette004_pos"], mat2euler(quat2mat(obs["pipette004_quat"])), obs["pipette004_quat"], flush=True)
    # print("🔱", "{:03}".format(n), obs["robot0_left_eef_pos"], obs["robot0_left_eef_quat"], obs["pipette004_quat"])
    print(env._gripper1_to_target_pos, env._gripper1_to_target_quat, env._pipette004_pos_bottom - (env._tube008_pos+env.object_offset))
    # print(np.linalg.norm(obs["g1_to_target_pos"]), obs["g1_to_target_quat"], obs["g1_to_target_pos"], flush=True)
    # print(obs["g0_to_target_pos"], np.linalg.norm(obs["g0_to_target_pos"]),obs["g0_to_target_quat"], flush=True)

    # pre_joint_positions = joint_positions
    # joint_positions = env.robots[0].sim.data.qpos
    # # print("body     :",np.array([joint_positions[0]]))
    # # print("left_arm :",joint_positions[1:9])
    # # print("right_arm:",joint_positions[10:18])
    # joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
    # delta_joint_positions = joint_positions - pre_joint_positions
    # action_seq_joint.append(delta_joint_positions)

    env.render()

print_joint_positions(env.robots[0].sim.data.qpos)

env.close()

# for key,value in obs.items():
#     print(f"Key: {key}, Value.shape: {value.shape}")
#     print(value)

# action_seq = np.array(action_seq)
# print("action_seq.shape: ",action_seq.shape)
# np.save("./collectdata/action_OSC/action_seq_OSC_"+str(args.episode)+".npy", action_seq)
# action_seq_joint = np.array(action_seq_joint)
# print("action_seq_joint.shape: ",action_seq_joint.shape)
# np.save("./collectdata/action_joint/action_seq_joint_"+str(args.episode)+".npy", action_seq_joint)

# obs_seq = np.array(obs_seq)
# print("obs_seq.shape: ",obs_seq.shape)
# np.save("./collectdata/obs/obs_seq_OSC_"+str(args.episode)+".npy", obs_seq)

# reward_seq = np.array(reward_seq)
# print("reward_seq.shape: ",reward_seq.shape)
# np.save("./collectdata/reward/reward_seq_OSC_"+str(args.episode)+".npy", reward_seq)




