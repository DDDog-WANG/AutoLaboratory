from math import pi, degrees
import numpy as np
import time
from tqdm import tqdm
import imageio
import argparse
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="action_seq_OSC2.0")
    parser.add_argument("--t", type=int, default=30)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()

writer = imageio.get_writer("../videos/"+args.video_name+".mp4", fps=60)

controller_config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    # use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
    initialization_noise=None
)
obs = env.reset()
for key,value in obs.items():
    print(f"Key: {key}, Value.shape: {value.shape}")
action = np.zeros(env.robots[0].dof)
action_seq = np.load("./collectdata/action_seq_OSC.npy")
# obs = env.reset()
for n in tqdm(range(args.t)):
    action = action_seq[n]
    obs, reward, done, _ = env.step(action)

    frame = obs[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
env.close()
writer.close()

# print("👑 env._get_observations(): ",dir(obs))
for key,value in obs.items():
    print(f"Key: {key}, Value.shape: {value.shape}")