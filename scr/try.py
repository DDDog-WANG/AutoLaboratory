from math import pi
from math import degrees
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
controller_config = load_controller_config(default_controller="JOINT_POSITION")
env = suite.make(
    env_name="MaholoLaboratory",
    robots="Maholo",
    gripper_types=["PandaGripper", "PandaGripper"],
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    control_freq=10,
    horizon = 100,
)
for key in env.robots[0].gripper:
    print(f"{key} hand: {env.robots[0].gripper[key]}")

obs = env.reset()
# action=np.zeros(17)
for ep in range(50):
    action=np.random.rand(17)
    obs, reward, done, _ = env.step(action)
    env.render()
env.close()

# for joint_index in env.robots[0].joint_indexes:
#     joint_name = env.sim.model.joint_id2name(joint_index)
#     print(f' "{joint_index}: {joint_name}" ', end="; ")

# for j in range(9):
#     env.reset()
#     action = np.zeros(17)
#     if j == 0: action[j]=1
#     elif j == 8: action[j], action[8+j] = 1, 1
#     elif j == 6: action[j], action[8+j] = 1, -1
#     elif j == 7: action[j], action[8+j] = 1, -1
#     else: action[j], action[8+j] = -1, 1
#     # action[j], action[8+j] = -1, 1
#     print(f"👑 joint_{j}_name: joint_names[j]")
#     for ep in range(10):
#         obs, reward, done, _ = env.step(action)
#         print("{:02}".format(ep), ["{:.4f}".format(round(x, 4)) for x in env.robots[0]._joint_positions])
#         # print(f"{ep}: {[round(x, 4) for x in env.robots[0]._joint_velocities]}")
#         env.render()

# print("👑 env: ",dir(env))
# print("👑 env.robots[0]: ",dir(env.robots[0]))

# print("👑 env._get_observations(): ",dir(obs))
# for key,value in obs.items():
#     print(f"Key: {key}, Value.shape: {value.shape}")