from math import pi, degrees
import argparse
import numpy as np
from tqdm import tqdm
import json
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
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--fix_initial_joint", type=bool, default=True)
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
)
env = GymWrapper(env) 
env = TimeFeatureWrapper(env)

obs_seq = []
reward_seq = []

listener = keyboard.Listener()
action = np.zeros(14)
action_seq = []
delta = 1
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

obs = env.reset()
# SET INITIAL JOINT POS
if args.fix_initial_joint:
    with open("./collectdata/initial_joint.json", "r") as file:
        initial_joint = json.load(file)
    for key,value in initial_joint.items():
            # print(f"Joint Name: {key}, Joint ID: {value}")
            env.sim.data.set_joint_qpos(key, value)
    env.sim.forward()


joint_names = env.sim.model.joint_names
joint_ids = [env.sim.model.joint_name2id(name) for name in joint_names]

eef_pos = env.sim.data.get_body_xpos("gripper0_left_eef")
eef_quat = env.sim.data.get_body_xquat("gripper0_left_eef")
eef_euler = mat2euler(quat2mat(eef_quat))
print(f"left_eef:  {eef_pos}, {eef_euler}")
eef_pos = env.sim.data.get_body_xpos("gripper0_right_eef")
eef_quat = env.sim.data.get_body_xquat("gripper0_right_eef")
eef_euler = mat2euler(quat2mat(eef_quat))
print(f"right_eef: {eef_pos}, {eef_euler}")
    
# joint_positions = env.robots[0].sim.data.qpos
# joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
for n in tqdm(range(args.horizon+1)):
    obs_seq.append(obs)
    action_seq.append(action.copy())
    obs, reward, done, _ = env.step(action)
    reward_seq.append(reward)

    # pre_joint_positions = joint_positions
    # joint_positions = env.robots[0].sim.data.qpos
    # joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
    # delta_joint_positions = joint_positions - pre_joint_positions
    # action_seq.append(delta_joint_positions)

    env.unwrapped.render()
env.close()

action_seq = np.array(action_seq)
print(action_seq.shape)
np.save("./collectdata/action_seq_OSC.npy", action_seq)

obs_seq = np.array(obs_seq)
print(obs_seq.shape)
np.save("./collectdata/obs_seq.npy", obs_seq)

reward_seq = np.array(reward_seq)
print(reward_seq.shape)
np.save("./collectdata/reward_seq.npy", reward_seq)



