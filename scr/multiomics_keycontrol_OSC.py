from math import pi
from math import degrees
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
from pynput import keyboard
controller_config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    env_name="MaholoLaboratory",
    robots="Maholo",
    gripper_types=["PandaGripper"],
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    control_freq=50,
    horizon = 10000,
    render_camera="frontview",
)

listener = keyboard.Listener()
action = np.zeros(14)
action_seq = []
delta = 1
key_li = ["0","1","2","3","4","5","6","7","8","~", "!",'"',"#","$","%","&","'","("]
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
        elif key.char == "f":
            action[3+7] = delta
        elif key.char == "h":
            action[3+7] = -delta
        elif key.char == "g":
            action[4+7] = delta
        elif key.char == "t":
            action[4+7] = -delta
        elif key.char == "r":
            action[5+7] = delta
        elif key.char == "y":
            action[5+7] = -delta
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
for n in range(100000):
    # print(action)
    action_seq.append(action)
    obs, reward, done, _ = env.step(action)
    env.render()
env.close()
action_seq = np.array(action_seq)
np.save("action_seq.npy", action_seq)
