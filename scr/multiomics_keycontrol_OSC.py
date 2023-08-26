from math import pi, degrees
import argparse
import numpy as np
from tqdm import tqdm
import json, time
from copy import deepcopy
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
from pynput import keyboard
from robosuite.wrappers.gym_wrapper import GymWrapper
from sb3_contrib.common.wrappers import TimeFeatureWrapper


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
env = GymWrapper(env) 
env = TimeFeatureWrapper(env)


delta = 0.6
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

        if key.char == "W":
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

obs = env.reset()

print(f"👑 ROUND {args.episode}")
eef_pos = env.sim.data.get_body_xpos("gripper0_left_eef")
eef_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("gripper0_left_eef")))
print(f"left_eef : {eef_pos}, {eef_euler}")
eef_pos = env.sim.data.get_body_xpos("gripper0_right_eef")
eef_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("gripper0_right_eef")))
print(f"right_eef: {eef_pos}, {eef_euler}")
pipette_pos = env.sim.data.get_body_xpos("P1000_withtip004_main")
pipette_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("P1000_withtip004_main")))
print(f"pipette  : {pipette_pos}, {pipette_euler}")
tube_pos = env.sim.data.get_body_xpos("tube1_5ml008_main")
tube_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("tube1_5ml008_main")))
print(f"tube     : {tube_pos}, {tube_euler}")

joint_positions = env.robots[0].sim.data.qpos
joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
time.sleep(1)

for n in tqdm(range(args.horizon)):

    obs_seq.append(obs)
    action_seq.append(action.copy())

    obs, reward, done, _ = env.step(action)
    reward_seq.append(reward)

    pre_joint_positions = joint_positions
    joint_positions = env.robots[0].sim.data.qpos
    joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
    delta_joint_positions = joint_positions - pre_joint_positions
    action_seq_joint.append(delta_joint_positions)

    env.unwrapped.render()
env.close()

action_seq = np.array(action_seq)
print("action_seq.shape: ",action_seq.shape)
np.save("./collectdata/action_seq_OSC_"+str(args.episode)+".npy", action_seq)
action_seq_joint = np.array(action_seq_joint)
print("action_seq_joint.shape: ",action_seq_joint.shape)
np.save("./collectdata/action_seq_joint_"+str(args.episode)+".npy", action_seq_joint)

obs_seq = np.array(obs_seq)
print("obs_seq.shape: ",obs_seq.shape)
np.save("./collectdata/obs_seq_OSC_"+str(args.episode)+".npy", obs_seq)

reward_seq = np.array(reward_seq)
print("reward_seq.shape: ",reward_seq.shape)
np.save("./collectdata/reward_seq_OSC_"+str(args.episode)+".npy", reward_seq)




