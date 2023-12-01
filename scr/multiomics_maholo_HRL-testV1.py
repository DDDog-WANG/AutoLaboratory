import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
import torch
from stable_baselines3 import DDPG , SAC, PPO
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import argparse
import imageio
import torch
# np.set_printoptions(precision=5, suppress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str)
    parser.add_argument("--model_load_eefR_Move2Pipette", type=str)
    parser.add_argument("--model_load_eefR_Grip2Pipette", type=str)
    parser.add_argument("--model_name", type=str, default="SAC")
    parser.add_argument("--policy", type=str, default="middle")

    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="rl_video")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()
controller_config = load_controller_config(default_controller=args.controller)
env_recorder = suite.make(
    "MaholoLaboratory_eefR_Move2Pipette",
    args.robots,
    robot_initial_qpos=np.zeros(15),
    gripper_r_initial_qpos = None,
    gripper_l_initial_qpos = None,
    tube008_initial_pos = np.array([0.066168, 0.29903, 0.955]),
    tube008_initial_quat = np.array([1, 0, 0, 0]),
    pipette004_initial_pos = np.array([-0.5897, -0.46573, 0.87555]),
    pipette004_initial_quat = np.array([0, -1, 0, 0]),
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
    initialization_noise=None,
)
env_eefR_Move2Pipette = suite.make(
    "MaholoLaboratory_eefR_Move2Pipette",
    args.robots,
    robot_initial_qpos=np.zeros(15),
    gripper_r_initial_qpos = None,
    gripper_l_initial_qpos = None,
    tube008_initial_pos = np.array([0.066168, 0.29903, 0.955]),
    tube008_initial_quat = np.array([1, 0, 0, 0]),
    pipette004_initial_pos = np.array([-0.5897, -0.46573, 0.87555]),
    pipette004_initial_quat = np.array([0, -1, 0, 0]),
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    render_gpu_device_id=0,
    horizon=args.horizon,
    initialization_noise=None
)
env_eefR_Move2Pipette = GymWrapper(env_eefR_Move2Pipette)

policy_kwargs = {'net_arch' : [512, 512, 512, 512], 
                'n_critics' : 4,
                }
model_eefR_Move2Pipette = SAC(policy="MlpPolicy", env=env_eefR_Move2Pipette, policy_kwargs=policy_kwargs)
model_eefR_Move2Pipette.policy.load_state_dict(torch.load(args.model_load_eefR_Move2Pipette))

writer = imageio.get_writer(args.workdir+"/videos_tmp/"+args.video_name+".mp4", fps=args.fps)

def print_joint_positions(joint_positions):
    print("body         :", joint_positions[0])
    print("left_arm     :", joint_positions[1:8])
    print("right_arm    :", joint_positions[10:17])
    print("left_gripper :", joint_positions[8:10])
    print("right_gripper:", joint_positions[17:19])
    print("pipette004_pos :", joint_positions[19:22])
    print("pipette004_quat:", joint_positions[22:26])
    print("tube008_pos    :", joint_positions[26:29])
    print("tube008_quat   :", joint_positions[29:33])
print(f"🚀 Agent MaholoLaboratory_eefR_Move2Pipette start working")
# env.robots[0].sim.data.qpos
print(f"👑 env_recorder.robots[0].sim.data.qpos.shape         : {env_recorder.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_recorder.robots[0].sim.data.qpos)
print(f"👑 env_eefR_Move2Pipette.robots[0].sim.data.qpos.shape: {env_eefR_Move2Pipette.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_eefR_Move2Pipette.robots[0].sim.data.qpos)
# env start simulating
rewards = 0
n = 1
obs = env_eefR_Move2Pipette.reset()
while not env_eefR_Move2Pipette.unwrapped._check_success() and n <= args.horizon/2:

    # Env for env_eefR_Move2Pipette
    action, _states = model_eefR_Move2Pipette.predict(obs, deterministic = True)
    obs, reward, done, _ = env_eefR_Move2Pipette.step(action)
    # Env for env_recorder
    obs_recorder, reward_recorder, _, _ = env_recorder.step(action)
    
    print("🔱", "{:03}".format(n), "Agent eefR_Move2Pipette", "{:.5f}".format(reward), flush=True)
    rewards += reward
    frame = obs_recorder[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
    n += 1
print(f"🔱 FINISH. rewards: {rewards}, steps: {n-1}, avg_rewards: {rewards/(n-1)}")
# env.robots[0].sim.data.qpos
print(f"👑 env_recorder.robots[0].sim.data.qpos.shape         : {env_recorder.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_recorder.robots[0].sim.data.qpos)
print(f"👑 env_eefR_Move2Pipette.robots[0].sim.data.qpos.shape: {env_eefR_Move2Pipette.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_eefR_Move2Pipette.robots[0].sim.data.qpos)
# obs_env
print("📷 obs_env_eefR_Move2Pipette: \n", obs)
print("✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢")
print("✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤")


joint_positions = env_eefR_Move2Pipette.robots[0].sim.data.qpos
env_eefR_Grip2Pipette = suite.make(
    "MaholoLaboratory_eefR_Grip2Pipette",
    args.robots,
    robot_initial_qpos=np.concatenate((joint_positions[:8],joint_positions[10:17])),
    gripper_r_initial_qpos=joint_positions[8:10],
    gripper_l_initial_qpos=joint_positions[17:19],
    pipette004_initial_pos=joint_positions[19:22],
    pipette004_initial_quat=joint_positions[22:26],
    tube008_initial_pos=joint_positions[26:29],
    tube008_initial_quat=joint_positions[29:33],
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    render_gpu_device_id=0,
    horizon=args.horizon,
    initialization_noise=None
)
env_eefR_Grip2Pipette = GymWrapper(env_eefR_Grip2Pipette)
model_eefR_Grip2Pipette = SAC(policy="MlpPolicy", env=env_eefR_Grip2Pipette, policy_kwargs=policy_kwargs)
model_eefR_Grip2Pipette.policy.load_state_dict(torch.load(args.model_load_eefR_Grip2Pipette))

print(f"🚀 Agent MaholoLaboratory_eefR_Grip2Pipette start working")
# env.robots[0].sim.data.qpos
print(f"👑 env_recorder.robots[0].sim.data.qpos.shape         : {env_recorder.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_recorder.robots[0].sim.data.qpos)
print(f"👑 env_eefR_Grip2Pipette.robots[0].sim.data.qpos.shape: {env_eefR_Grip2Pipette.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_eefR_Grip2Pipette.robots[0].sim.data.qpos)
# obs_env
obs = env_eefR_Grip2Pipette.reset()
print("📷 obs_env_eefR_Grip2Pipette:         ", obs)
# env start simulating
rewards = 0
n = 1
while not env_eefR_Grip2Pipette.unwrapped._check_success() and n <= args.horizon/2:

    # Env for eefR_Grip2Pipette
    action, _states = model_eefR_Grip2Pipette.predict(obs, deterministic = True)
    obs, reward, done, _ = env_eefR_Grip2Pipette.step(action)
    # Env for recorder
    obs_recorder, reward_recorder, _, _ = env_recorder.step(action)

    print("🔱", "{:03}".format(n), "Agent eefR_Grip2Pipette", "{:.5f}".format(reward), flush=True)
    rewards += reward
    frame = obs_recorder[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
    n += 1
print(f"🔱 FINISH. rewards: {rewards}, steps: {n-1}, avg_rewards: {rewards/(n-1)}")
# env.robots[0].sim.data.qpos
print(f"👑 env_recorder.robots[0].sim.data.qpos.shape         : {env_recorder.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_recorder.robots[0].sim.data.qpos)
print(f"👑 env_eefR_Grip2Pipette.robots[0].sim.data.qpos.shape: {env_eefR_Grip2Pipette.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_eefR_Grip2Pipette.robots[0].sim.data.qpos)
# obs_env
print("📷 obs_env_eefR_Grip2Pipette: \n", obs)
print("✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢")
print("✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤")




env_eefR_Move2Pipette.close()
env_eefR_Grip2Pipette.close()
env_recorder.close()
writer.close()