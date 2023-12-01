import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
import numpy as np
from stable_baselines3 import DDPG , SAC, PPO
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import argparse
import imageio
import torch
np.set_printoptions(precision=5, suppress=True)

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
    "MaholoLaboratory_HRL",
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
    initialization_noise=None,
)
env_agent = suite.make(
    "MaholoLaboratory_HRL",
    args.robots,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    render_gpu_device_id=0,
    horizon=args.horizon,
    initialization_noise=None,
    task="move2pipette004"
)
env_agent = TimeFeatureWrapper(GymWrapper(env_agent))

policy_kwargs = {'net_arch' : [512, 512, 512, 512], 
                'n_critics' : 4,
                }
model_eefR_Move2Pipette = SAC(policy="MlpPolicy", env=env_agent, policy_kwargs=policy_kwargs)
model_eefR_Move2Pipette.policy.load_state_dict(torch.load(args.model_load_eefR_Move2Pipette))
model_eefR_Grip2Pipette = SAC(policy="MlpPolicy", env=env_agent, policy_kwargs=policy_kwargs)
model_eefR_Grip2Pipette.policy.load_state_dict(torch.load(args.model_load_eefR_Grip2Pipette))

writer = imageio.get_writer(args.workdir+"/videos_tmp/"+args.video_name+".mp4", fps=args.fps)

def print_joint_positions(joint_positions):
    print("body         :", joint_positions[0])
    print("left_arm     :", joint_positions[1:8])
    print("left_gripper :", joint_positions[8:10])
    print("right_arm    :", joint_positions[10:17])
    print("right_gripper:", joint_positions[17:19])
    print("pipette004_pos :", joint_positions[19:22])
    print("pipette004_quat:", joint_positions[22:26])
    print("tube008_pos    :", joint_positions[26:29])
    print("tube008_quat   :", joint_positions[29:33])
print(f"🚀 Agent MaholoLaboratory_eefR_Move2Pipette start working")
# env.robots[0].sim.data.qpos
print(f"👑 env_recorder.robots[0].sim.data.qpos.shape         : {env_recorder.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_recorder.robots[0].sim.data.qpos)
print(f"👑 env_agent.robots[0].sim.data.qpos.shape: {env_agent.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_agent.robots[0].sim.data.qpos)
# env start simulating
obs = env_agent.reset()
for n in range(args.horizon):
    
    if not env_agent.unwrapped._check_success_eefR_Move2Pipette():
        action, _states = model_eefR_Move2Pipette.predict(obs, deterministic = True)
        print("🔱", "{:03}".format(n), "Agent eefR_Move2Pipette", end=" ", flush=True)
    elif not env_agent.unwrapped._check_success_eefR_Grip2Pipette():
        action, _states = model_eefR_Grip2Pipette.predict(obs, deterministic = True)
        print("🔱", "{:03}".format(n), "Agent eefR_Grip2Pipette", end=" ", flush=True)
    else:
        action = np.zeros(17)
        print("🔱", "{:03}".format(n), "Agent None", end=" ", flush=True)

    obs, reward, done, _ = env_agent.step(action)
    obs_recorder, _, _, _ = env_recorder.step(action)
    print(np.linalg.norm(obs_recorder["g1_to_target_pos"]), obs_recorder["g1_to_target_quat"], env_agent.unwrapped._check_success_eefR_Move2Pipette(), env_agent.unwrapped._check_success_eefR_Grip2Pipette(), flush=True)
    frame = obs_recorder[args.camera+"_image"]
    frame = np.flip(frame, axis=0)
    writer.append_data(frame)
print(f"🔱 FINISH")
# env.robots[0].sim.data.qpos
print(f"👑 env_recorder.robots[0].sim.data.qpos.shape         : {env_recorder.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_recorder.robots[0].sim.data.qpos)
print(f"👑 env_agent.robots[0].sim.data.qpos.shape: {env_agent.robots[0].sim.data.qpos.shape}")
print_joint_positions(env_agent.robots[0].sim.data.qpos)
print("✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢")
print("✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤")



env_recorder.close()
env_agent.close()
writer.close()