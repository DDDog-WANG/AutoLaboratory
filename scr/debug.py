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
np.set_printoptions(precision=4, suppress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory_eefR_Move2Tube")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()


controller_config = load_controller_config(default_controller=args.controller)
env = suite.make(
    args.environment,
    args.robots,
    # gripper_types=None,
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_object_obs=False,
    use_camera_obs=True,
    control_freq=20,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
    horizon=args.horizon,
    initialization_noise=None,
    robot_initial_qpos = np.ones(15)
)
# env = GymWrapper(env) 
# env = TimeFeatureWrapper(env)

for key in env.robots[0].gripper:
    print(f"{key} hand: {env.robots[0].gripper[key]}")
# action_seq = np.load("./collectdata/action_OSC/action_seq_OSC_1.npy")
# action_seq_joint = []


env.reset()
joint_positions = env.robots[0].sim.data.qpos
joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
# joint_positions = joint_positions[:8]

print("✤✤ Object ✤✤")
pipette_pos = env.sim.data.get_body_xpos("P1000_withtip004_main")
pipette_xquat = env.sim.data.get_body_xquat("P1000_withtip004_main")
print(f'pipette  : {pipette_pos}, {pipette_xquat}')
tube_pos = env.sim.data.get_body_xpos("tube1_5ml008_main")
tube_xquat = env.sim.data.get_body_xquat("tube1_5ml008_main")
print(f'tube     : {tube_pos}, {tube_xquat}')

print("✤✤ Robot ✤✤")
joint_positions = env.robots[0].sim.data.qpos
print(joint_positions.shape)
print("body         :", joint_positions[0])
print("left_arm     :", joint_positions[1:8])
print("left_gripper :", joint_positions[8:10])
print("right_arm    :", joint_positions[10:17])
print("right_gripper:", joint_positions[17:19])
print("pipette004_pos :", joint_positions[19:22])
print("pipette004_quat:", joint_positions[22:26])
print("tube008_pos    :", joint_positions[26:29])
print("tube008_quat   :", joint_positions[29:33])

eef_pos = env.sim.data.get_body_xpos("gripper0_left_eef")
eef_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("gripper0_left_eef")))
print(f'left_eef : {eef_pos}, {env.sim.data.get_body_xquat("gripper0_left_eef")}')
eef_pos = env.sim.data.get_body_xpos("gripper0_right_eef")
eef_euler = mat2euler(quat2mat(env.sim.data.get_body_xquat("gripper0_right_eef")))
print(f'right_eef: {eef_pos}, {env.sim.data.get_body_xquat("gripper0_right_eef")}')

print(env.robots[0].gripper["right"])
print(dir(env.robots[0].gripper["right"]))
print(env.robots[0].gripper_joints["right"])
print(dir(env.robots[0].gripper_joints["right"]))

print('env.robots[0].gripper["right"].init_qpos: ', env.robots[0].gripper["right"].init_qpos)
print('env.robots[0].gripper["left"].init_qpos : ', env.robots[0].gripper["left"].init_qpos)
print("env.robots[0].sim.data.qpos[:19]: ", env.robots[0].sim.data.qpos[:19])
print("env.robots[0].gripper['right'].gripper_init_qpos: ", env.robots[0].gripper["right"].gripper_init_qpos)
print("env.robots[0].gripper['left'].gripper_init_qpos : ", env.robots[0].gripper["left"].gripper_init_qpos)
print("✣✣✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢✢")
print("✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤✤")

obs = env.reset()
for key,value in obs.items():
    print(f"Key: {key}, Value.shape: {value.shape}")

action = np.zeros(17)
# action[16] = -1
for n in range(100):
    # action = np.random.uniform(-1, 1, size=env.robots[0].dof)
    # action = action_seq[n]
    # action = np.array([action[0]/0.0034, 
    #                   action[1]/0.0033, action[2]/0.0032, action[3]/0.0032, action[4]/0.0033, action[5]/0.0031, action[6]/0.0029, action[7]/0.0029, action[8]/0.0001, 
    #                   action[9]/0.0032, action[10]/0.0031, action[11]/0.0032, action[12]/0.0033, action[13]/0.0029, action[14]/0.0027, action[15]/0.0026, action[16]/0.0001])
    # print(f"{n:03} {action}")

    # pre_joint_positions = joint_positions.copy()
    obs, reward, done, _ = env.step(action)
    # print("🔱", "{:03}".format(n), env.robots[0].sim.data.qpos[17:19], flush=True)
    # print(action)
    # print("🔱", "{:03}".format(n), "{:.5f}".format(reward), flush=True)
    # print(np.linalg.norm(obs["g1_to_target_pos"]), obs["g1_to_target_quat"], obs["g1_to_target_pos"], flush=True)
    # print(obs["pipette004_pos"], obs["pipette004_quat"], flush=True)
    # print(obs["gripper1_to_pipette004"], obs["gripper1_quat_pipette004"], flush=True)
    # joint_positions = env.robots[0].sim.data.qpos
    # joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
    # # joint_positions = joint_positions[:8]

    # delta_joint_positions = joint_positions - pre_joint_positions
    # action_seq_joint.append(delta_joint_positions)
    # print(action/delta_joint_positions)

    # cube_pos = env.sim.data.get_body_xpos("cube_main")
    # cube_quat = env.sim.data.get_body_xquat("cube_main")
    # cube_euler = mat2euler(quat2mat(cube_quat))
    # print("cube: ",cube_pos, cube_euler)
    # env.render()
    # env.unwrapped.render()

env.close()


# for key,value in obs.items():
#     print(f"Key: {key}, Value.shape: {value.shape}")
#     print(type(value))

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
# print("👑 obs: ",dir(obs))
# for key,value in obs.items():
#     print(f"🟡 Key: {key}, Value.shape: {value.shape}")
#     print(value)


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