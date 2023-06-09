from math import pi
from math import degrees
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
from pynput import keyboard
controller_config = load_controller_config(default_controller="JOINT_POSITION")
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
action = np.zeros(env.robots[0].dof)
action_seq = []
delta = 5
key_li = ["0","1","2","3","4","5","6","7","8","~", "!",'"',"#","$","%","&","'","("]
def on_press(key):
    try:
        for i in range(len(key_li)):
            if i == 0:
                if key.char == key_li[i]:
                    action[0] = delta
            elif i <= 8:
                if key.char == key_li[i]:
                    action[i+8] = delta
            elif i == 9:
                if key.char == key_li[i]:
                    action[0] = -delta
            elif i <= 17:
                if key.char == key_li[i]:
                    action[i-1] = -delta
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
for n in range(10000):
    # print(action)
    action_seq.append(action)
    obs, reward, done, _ = env.step(action)
    env.render()
env.close()
action_seq = np.array(action_seq)
np.save("action_seq.npy", action_seq)
