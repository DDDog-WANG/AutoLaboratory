from math import pi, degrees
import argparse
import numpy as np
from tqdm import tqdm
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
)

env = GymWrapper(env) 
env = TimeFeatureWrapper(env)

action = np.zeros(env.robots[0].dof)
action_seq = []
obs_seq = []
reward_seq = []

listener = keyboard.Listener()
action = np.zeros(14)
action_seq = []
delta = 1
def on_press(key):
    try:
        if key.char == "w":
            action[0+7] = -delta
        elif key.char == "s":
            action[0+7] = delta
        elif key.char == "a":
            action[1+7] = -delta
        elif key.char == "d":
            action[1+7] = delta
        elif key.char == "q":
            action[2+7] = delta
        elif key.char == "e":
            action[2+7] = -delta

        elif key.char == "j":
            action[3+7] = delta/2
        elif key.char == "l":
            action[3+7] = -delta/2
        elif key.char == "k":
            action[4+7] = delta/2
        elif key.char == "i":
            action[4+7] = -delta/2
        elif key.char == "o":
            action[5+7] = delta/2
        elif key.char == "u":
            action[5+7] = -delta/2
        elif key.char == "1":
            action[6+7] = delta
        elif key.char == "0":
            action[6+7] = -delta
    except AttributeError:
        pass
def on_release(key):
    try:
        action[:] = 0
    except AttributeError:
        pass
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

obs = env.reset()
for n in range(args.horizon):
    obs_seq.append(obs)
    joint_action = env.sim.data.ctrl.copy()
    print(joint_action)
    action_seq.append(joint_action)

    obs, reward, done, _ = env.step(action)
    reward_seq.append(reward)

    env.unwrapped.render()
env.close()

action_seq = np.array(action_seq)
print(action_seq.shape)
np.save("./collectdata/action_seq.npy", action_seq)

obs_seq = np.array(obs_seq)
print(obs_seq.shape)
np.save("./collectdata/obs_seq.npy", obs_seq)

reward_seq = np.array(reward_seq)
print(reward_seq.shape)
np.save("./collectdata/reward_seq.npy", reward_seq)

actuator_names = env.robots[0].sim.model.actuator_names
print(actuator_names)


