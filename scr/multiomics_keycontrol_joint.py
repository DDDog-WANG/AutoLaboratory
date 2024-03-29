from math import pi, degrees
import argparse
import numpy as np
from tqdm import tqdm
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler
from pynput import keyboard
from robosuite.wrappers.gym_wrapper import GymWrapper
from sb3_contrib.common.wrappers import TimeFeatureWrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="MaholoLaboratory_eefR_Move2Pipette")
    parser.add_argument("--robots", type=str, default="Maholo")
    parser.add_argument("--camera", type=str, default="frontview")
    parser.add_argument("--video_name", type=str, default="my_video")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--width", type=int, default=2560)
    args = parser.parse_args()

controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
env = suite.make(
    args.environment,
    args.robots,
    gripper_types=["PandaGripper"],
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    control_freq=50,
    render_camera=args.camera,
    camera_names=args.camera,
    camera_heights=args.height,
    camera_widths=args.width,
    initialization_noise=None
)
env = GymWrapper(env) 
env = TimeFeatureWrapper(env)

action = np.zeros(env.robots[0].dof)
action_seq = []
obs_seq = []
reward_seq = []

delta = 1
key_li = ["0","1","2","3","4","5","6","7","8","~", "!",'"',"#","$","%","&","'","("]
def on_press(key):
    global action
    try:
        for i in range(len(key_li)):
            if i == 0:
                if key.char == key_li[i]:
                    action[0] = delta
            elif i <= 8:
                if key.char == key_li[i]:
                    action[i+8] = delta
            elif i == 9:
                if key.char == key_li[i]:
                    action[0] = -delta
            elif i <= 17:
                if key.char == key_li[i]:
                    action[i-1] = -delta
    except AttributeError:
        pass
def on_release(key):
    global action
    try:
        action[:] = 0
    except AttributeError:
        pass
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

obs = env.reset()

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

# joint_names = env.sim.model.joint_names
# joint_ids = [env.sim.model.joint_name2id(name) for name in joint_names] 
# joint_positions = env.robots[0].sim.data.qpos
# joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))

for n in range(args.horizon):
    obs_seq.append(obs)
    action_seq.append(action.copy())

    obs, reward, done, _ = env.step(action)
    # print("🔱", "{:03}".format(n), "{:.5f}".format(reward), flush=True)
    print(action)
    reward_seq.append(reward)

    # pre_joint_positions = joint_positions
    # joint_positions = env.robots[0].sim.data.qpos
    # joint_positions = np.concatenate((joint_positions[:9],joint_positions[10:18]))
    # delta_joint_positions = joint_positions - pre_joint_positions
    # action_seq.append(delta_joint_positions)

    env.unwrapped.render()
env.close()

# action_seq = np.array(action_seq)
# print(action_seq.shape)
# np.save("./collectdata/action_seq_joint.npy", action_seq)

# obs_seq = np.array(obs_seq)
# print(obs_seq.shape)
# np.save("./collectdata/obs_seq_joint.npy", obs_seq)

# reward_seq = np.array(reward_seq)
# print(reward_seq.shape)
# np.save("./collectdata/reward_seq_joint.npy", reward_seq)

