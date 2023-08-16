import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
from stable_baselines3 import DDPG , SAC, PPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, help="work directory")
    parser.add_argument("--model_save", type=str, help="save to directory")
    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)

    parser.add_argument("--model_name", type=str, default="DDPG")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

env = suite.make(
    args.environment,
    args.robots,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    render_gpu_device_id=0,
    horizon=args.horizon,
)
for key,value in env.reset().items():
    print(f"Key: {key}, Value.shape: {value.shape}", flush=True)
env = GymWrapper(env)
env = TimeFeatureWrapper(env)
print(f"\nTimeFeature GYM Wrapper obs.shape: {env.reset().shape}\n", flush=True)

batch_size = args.batch_size
learning_rate = args.learning_rate
total_timesteps = args.horizon * args.episodes
policy_kwargs = {'net_arch' : [512, 512, 512, 512], 
                'n_critics' : 4,
                }
n_actions = env.robots[0].action_dim
print(f"n_actions: {n_actions}\n", flush=True)
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2)
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), theta=0.1, sigma=0.2)

if args.model_name == "DDPG":
    model = DDPG(policy="MlpPolicy", env=env, replay_buffer_class=ReplayBuffer, verbose=1, gamma=0.95, batch_size=batch_size, 
                buffer_size=100000, learning_rate=learning_rate, action_noise=action_noise, policy_kwargs=policy_kwargs)
elif args.model_name == "SAC":
    model = SAC(policy="MlpPolicy", env=env, replay_buffer_class=ReplayBuffer, verbose=1, gamma = 0.95, batch_size=batch_size, 
                buffer_size=100000, learning_rate=learning_rate, action_noise=action_noise, policy_kwargs=policy_kwargs)
elif args.model_name == "PPO":
    model = PPO(policy="MlpPolicy", env=env, verbose=1, gamma=0.95, batch_size=batch_size)
# model = SAC.load(workdir+'/models/SAC_big', env = env, learning_rate = learning_rate, action_noise = action_noise)
model.learn(total_timesteps=total_timesteps)
model.save(args.model_save)
print("Saved to ", args.model_save, flush=True)
