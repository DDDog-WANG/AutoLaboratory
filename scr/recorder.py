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
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()

writer = imageio.get_writer("../videos/"+args.video_name+".mp4", fps=100)

controller_config = load_controller_config(default_controller="JOINT_POSITION")
env = suite.make(
    args.environment,
    args.robots,
    # gripper_types=["PandaGripper"],
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    control_freq=100,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
)

action = np.zeros(env.robots[0].dof)

# obs = env.reset()
for n in tqdm(range(args.timesteps)):
    obs, reward, done, _ = env.step(action)

    frame = obs[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
env.close()
writer.close()