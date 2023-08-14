from math import pi, degrees
import argparse
import numpy as np
from tqdm import tqdm
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
from pynput import keyboard

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--t", type=int, default=10000)
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
    control_freq=50,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
)

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
for n in tqdm(range(args.t)):
    # print(action)
    action_seq.append(action)
    obs, reward, done, _ = env.step(action)
    print(reward)
    env.render()
env.close()
action_seq = np.array(action_seq)
np.save("action_seq.npy", action_seq)
