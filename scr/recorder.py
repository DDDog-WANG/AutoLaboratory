from math import pi, degrees
import numpy as np
import time
from tqdm import tqdm
import imageio
import argparse
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
from multiomics_maholo_move import maholo_move
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory_eefR_Move2Pipette")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="JOINT_VELOCITY")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="video")
    parser.add_argument("--t", type=int, default=50)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()

writer = imageio.get_writer("./videos/"+args.environment+"_"+args.controller+".mp4", fps=50)

controller_config = load_controller_config(default_controller=args.controller)
env = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    control_freq=50,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
    initialization_noise=None
)
obs = env.reset()
# for key,value in obs.items():
#     print(f"Key: {key}, Value.shape: {value.shape}")

# action_seq = np.load("./collectdata/action_seq_total_OSC.npy")
obs = env.reset()
dim = 14
# dim = env.robots[0].dof
action = np.zeros(dim)
for i in range(dim):
    action = np.zeros(dim)
    action[i] = 1
    print(i)
    for n in range(args.t):
        # action = action_seq[n]
        # action = maholo_move(obs)
        obs, reward, done, _ = env.step(action)

        frame = obs[args.camera+"_image"]
        frame = np.flip(frame, axis=0)
        writer.append_data(frame)
env.close()
writer.close()

# print("👑 env._get_observations(): ",dir(obs))
# for key,value in obs.items():
#     print(f"Key: {key}, Value.shape: {value.shape}")