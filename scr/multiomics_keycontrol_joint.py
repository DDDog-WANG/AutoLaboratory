from math import pi, degrees
import argparse
import numpy as np
from tqdm import tqdm
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
# from pynput import keyboard
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

controller_config = load_controller_config(default_controller="JOINT_POSITION")
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

# env = suite.make(
#     args.environment,
#     args.robots,
#     has_renderer=False,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
#     control_freq=50,
#     render_camera=args.camera,
#     render_gpu_device_id=0,
#     horizon=args.horizon,
# )
env = GymWrapper(env) 
env = TimeFeatureWrapper(env)

action = np.zeros(env.robots[0].dof)
action = np.random.uniform(-1, 1, env.robots[0].dof)
action_seq = []
obs_seq = []
reward_seq = []

# delta = 5
# key_li = ["0","1","2","3","4","5","6","7","8","~", "!",'"',"#","$","%","&","'","("]
# listener = keyboard.Listener()
# def on_press(key):
#     try:
#         for i in range(len(key_li)):
#             if i == 0:
#                 if key.char == key_li[i]:
#                     action[0] = delta
#             elif i <= 8:
#                 if key.char == key_li[i]:
#                     action[i+8] = delta
#             elif i == 9:
#                 if key.char == key_li[i]:
#                     action[0] = -delta
#             elif i <= 17:
#                 if key.char == key_li[i]:
#                     action[i-1] = -delta
#     except AttributeError:
#         pass

# def on_release(key):
#     try:
#         action[:] = 0
#     except AttributeError:
#         pass

# listener = keyboard.Listener(on_press=on_press, on_release=on_release)
# listener.start()
obs = env.reset()
for n in tqdm(range(args.horizon)):
    obs_seq.append(obs)
    action_seq.append(action)

    obs, reward, done, _ = env.step(action)
    # obs_wrap, reward_wrap, _, _ = env.step(action)
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

