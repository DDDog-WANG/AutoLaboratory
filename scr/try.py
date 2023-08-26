from math import pi, degrees
import numpy as np
import time
import json
from tqdm import tqdm
import argparse
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
from robosuite.wrappers.gym_wrapper import GymWrapper
from sb3_contrib.common.wrappers import TimeFeatureWrapper
np.set_printoptions(precision=6, suppress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()


controller_config = load_controller_config(default_controller="OSC_POSE")
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
    horizon=args.horizon,
    initialization_noise=None
)
env = GymWrapper(env) 
env = TimeFeatureWrapper(env)

# for key in env.robots[0].gripper:
#     print(f"{key} hand: {env.robots[0].gripper[key]}")
action_np = np.load("./collectdata/action_seq_OSC.npy")
action_seq_joint = []
# joint_positions = env.robots[0].sim.data.qpos
# joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
# action = np.ones(env.robots[0].dof)


obs = env.reset()
joint_names = env.sim.model.joint_names
joint_ids = [env.sim.model.joint_name2id(name) for name in joint_names]
joint_positions = env.robots[0].sim.data.qpos
joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))

for ep in range(args.episode):
    print(f"👑 ROUND {ep}")
    env.reset()
    eef_pos = env.sim.data.get_body_xpos("gripper0_left_eef")
    eef_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("gripper0_left_eef")))
    print(f"left_eef : {eef_pos}, {eef_euler}")
    eef_pos = env.sim.data.get_body_xpos("gripper0_right_eef")
    eef_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("gripper0_right_eef")))
    print(f"right_eef: {eef_pos}, {eef_euler}")
    pipette_pos = env.sim.data.get_body_xpos("P1000_withtip004_main")
    pipette_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("P1000_withtip004_main")))
    print(f"pipette  : {pipette_pos}, {pipette_euler}")
    tube_pos = env.sim.data.get_body_xpos("tube1_5ml008_main")
    tube_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("tube1_5ml008_main")))
    print(f"tube     : {tube_pos}, {tube_euler}")


    for n in tqdm(range(args.horizon)):

        # eef_pos = env.sim.data.get_body_xpos("gripper0_left_eef")
        # eef_quat = env.sim.data.get_body_xquat("gripper0_left_eef")
        # eef_euler = mat2euler(quat2mat(eef_quat))
        # print(f"{n:03} left_eef:  {eef_pos}, {eef_euler}")
        # eef_pos = env.sim.data.get_body_xpos("gripper0_right_eef")
        # eef_quat = env.sim.data.get_body_xquat("gripper0_right_eef")
        # eef_euler = mat2euler(quat2mat(eef_quat))
        # print(f"{n:03} right_eef: {eef_pos}, {eef_euler}")

        # eef_pos = env.sim.data.get_body_xpos("gripper0_eef")
        # eef_quat = env.sim.data.get_body_xquat("gripper0_eef")
        # eef_euler = mat2euler(quat2mat(eef_quat))
        # print(f"eef:  {eef_pos}, {eef_euler}")

        # action = np.zeros(14)
        # action = np.random.uniform(-1, 1, size=env.robots[0].dof)
        action = action_np[n]
        # print(action)

        obs, reward, done, _ = env.step(action)

        # pre_joint_positions = joint_positions
        # joint_positions = env.robots[0].sim.data.qpos
        # joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
        # delta_joint_positions = joint_positions - pre_joint_positions
        # action_seq_joint.append(delta_joint_positions)
        # print(delta_joint_positions)

        # cube_pos = env.sim.data.get_body_xpos("cube_main")
        # cube_quat = env.sim.data.get_body_xquat("cube_main")
        # cube_euler = mat2euler(quat2mat(cube_quat))
        # print("cube: ",cube_pos, cube_euler)

        env.unwrapped.render()
env.unwrapped.close()


# # SAVE INITIAL JOINT POS
# initial_joint = {}
# joint_names = env.sim.model.joint_names
# joint_ids = [env.sim.model.joint_name2id(name) for name in joint_names]

# for name, id in zip(joint_names, joint_ids): 
#     joint_pos = env.sim.data.get_joint_qpos(name)
#     print(f"Joint Name: {name}, Joint ID: {id}, Joint pos: {joint_pos}")
#     initial_joint[name] = joint_pos.tolist()
# print(initial_joint)
# with open("./collectdata/initial_joint.json", "w") as file:
#     json.dump(initial_joint, file)

# joint_positions = env.robots[0].sim.data.qpos
# print(joint_positions)




# print(action)

# joint_positions = env.robots[0].sim.data.qpos
# print(joint_positions[:8])
# print(joint_positions[8:10])
# print(joint_positions[10:17])
# print(joint_positions[17:19])







# print("👑 env: ",dir(env))
# print("👑 env.robots[0]: ",dir(env.robots[0]))
# print("👑 env._get_observations(): ",dir(obs))
# for key,value in obs.items():
#     print(f"🟡 Key: {key}, Value.shape: {value.shape}")
#     print(type(value))


# # SAVE INITIAL JOINT POS
# initial_joint = {}
# for name, id in zip(joint_names, joint_ids): 
#     joint_pos = env.sim.data.get_joint_qpos(name)
#     print(f"Joint: {name}, ID: {id}, pos: {joint_pos}")
#     initial_joint[name] = joint_pos.tolist()
# with open("./collectdata/initial_joint.json", "w") as file:
#     json.dump(initial_joint, file)

# # SET INITIAL JOINT POS
# with open("./collectdata/initial_joint.json", "r") as file:
#     initial_joint = json.load(file)
# if args.fix_initial_joint=="True":
#     for key,value in initial_joint.items():
#         # print(f"Joint Name: {key}, Joint ID: {value}")
#         env.sim.data.set_joint_qpos(key, value)
#     env.sim.forward()