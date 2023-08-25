import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
from stable_baselines3 import DDPG , SAC, PPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import argparse
import imageio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str)
    parser.add_argument("--model_save", type=str)
    parser.add_argument("--log_save", type=str)
    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="rl_video")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)

    parser.add_argument("--model_name", type=str, default="DDPG")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

controller_config = load_controller_config(default_controller=args.controller)
env_recoder = suite.make(
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
    render_gpu_device_id=0,
    horizon=args.horizon,
)
env = suite.make(
    args.environment,
    args.robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    render_gpu_device_id=0,
    horizon=args.horizon,
)
env = GymWrapper(env)
env = TimeFeatureWrapper(env)

writer = imageio.get_writer(args.workdir+"/videos/"+args.video_name+".mp4", fps=args.fps)
if args.model_name == "DDPG":
    model = DDPG.load(args.model_save)
elif args.model_name == "SAC":
    model = SAC.load(args.model_save)
elif args.model_name == "PPO":
    model = PPO.load(args.model_save)

obs = env.reset()
for n in range(args.horizon):
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, _ = env.step(action)
    obs_recoder, _, _, _ = env_recoder.step(action)
    # print("🔱", "{:03}".format(n), ["{:.4f}".format(x) for x in action], "{:.5f}".format(reward))
    # env.unwrapped.render()
    frame = obs_recoder[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
env.close()
env_recoder.close()
writer.close()
print("🔱 FINISH")
