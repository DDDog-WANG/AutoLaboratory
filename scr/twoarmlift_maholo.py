from math import pi
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler

controller_config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    env_name="TwoArmLift",
    robots="Maholo",
    gripper_types=["PandaGripper", "PandaGripper"],
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    control_freq=50,
    horizon = 500,
)

fac_tran = 20
fac_rot = 2
zeros = np.zeros(3)
def move(delta_pos, grip):
    action = np.concatenate((zeros, zeros, np.array([grip]) ))
    done = False
    if abs(delta_pos[0])>0.01 or abs(delta_pos[1])>0.01 or -delta_pos[2]>0.01:
        action = np.concatenate((delta_pos*fac_tran, zeros, np.array([grip]) ))
    else: done = True
    return action, done

def roll(delta_euler, grip):
    delta_euler[:3] = (delta_euler[:3] + pi) % (2 * pi) - pi
    # delta_euler[2] = (delta_euler[2] + pi/2) % (2 * pi/2) - pi/2
    action = np.concatenate((zeros, zeros, np.array([grip]) ))
    # done = False
    # if   abs(delta_euler[0]) > 0.02: action[0+3] = delta_euler[0]*5
    # elif abs(delta_euler[1]) > 0.02: action[1+3] = delta_euler[1]*5
    # elif abs(delta_euler[2]) > 0.02: action[2+3] = delta_euler[2]*5
    # else: done = True
    done = True
    max_idx = np.argmax(np.abs(delta_euler))
    if abs(delta_euler[max_idx]) > 0.02:
        action[max_idx + 3] = delta_euler[max_idx]*fac_rot
        done = False
    return action, done

done = done_roll = done_pick = done_move = done_down = False
step_pick = step_up = 0
epoches = 1
env.reset()
for i in range(epoches):
    env.reset()
    total_reward = 0.
    obs = env._get_observations()
    while not done:
        # robot_eef_info
        eef_left_pos = obs['robot0_left_eef_pos']
        eef_left_quat = obs['robot0_left_eef_quat']
        eef_left_euler = mat2euler(quat2mat(eef_left_quat))
        eef_right_pos = obs['robot0_right_eef_pos']
        eef_right_quat = obs['robot0_right_eef_quat']
        eef_right_euler = mat2euler(quat2mat(eef_right_quat))

        # pot_info handle0_xpos handle1_xpos
        pot_pos = obs['pot_pos']
        pot_quat = obs['pot_quat']
        pot_euler = mat2euler(quat2mat(pot_quat))
        handle0_pos = obs["handle0_xpos"]
        handle1_pos = obs["handle1_xpos"]

        # delta
        delta_left_pos = handle0_pos - eef_left_pos
        delta_right_pos = handle1_pos - eef_right_pos
        delta_up_left_pos = np.array([delta_left_pos[0], delta_left_pos[1], delta_left_pos[2]+0.1])
        delta_up_right_pos = np.array([delta_right_pos[0], delta_right_pos[1], delta_right_pos[2]+0.1])

        delta_right_euler = pot_euler - eef_right_euler
        delta_left_euler = pot_euler - eef_left_euler
        delta_right_euler = np.array([-delta_right_euler[0]+pi, -delta_right_euler[1], delta_right_euler[2]])
        delta_left_euler = np.array([-delta_left_euler[0]+pi, -delta_left_euler[1], delta_left_euler[2]])

        # action step
        action_left_roll, done_left_roll = roll(delta_left_euler, -1)
        action_right_roll, done_right_roll = roll(delta_right_euler, -1)
        action_left_move, done_left_move = move(delta_up_left_pos, -1)
        action_right_move, done_right_move = move(delta_up_right_pos, -1)
        action_left_down, done_left_down = move(delta_left_pos, -1)
        action_right_down, done_right_down = move(delta_right_pos, -1)

        done_roll = done_left_roll and done_right_roll
        done_move = done_left_move and done_right_move
        done_down = done_left_down and done_right_down

        if not done_roll:
            action = np.concatenate((action_right_roll, action_left_roll))
            print("🤡 [ROLL] ",[round(x/fac_rot, 4) for x in action])
        elif done_roll and not done_move:
            action = np.concatenate((action_right_move, action_left_move))
            print("👾 [MOVE] ",[round(x/fac_tran, 4) for x in action])
        elif done_roll and done_move and not done_down:
            action = np.concatenate((action_right_down, action_left_down))
            print("🎃 [DOWN] ",[round(x/fac_tran, 4) for x in action])
        elif done_roll and done_move and done_down and not done_pick:
            action = np.concatenate((zeros, zeros, np.array([1]), zeros, zeros, np.array([1]) ))
            print("👹 [PICK] ",[round(x, 4) for x in action])
            step_pick += 1
            if step_pick > 20: done_pick = True
        elif done_roll and done_move and done_down and done_pick:
            action =  np.concatenate((np.array([0, 0, 1]), zeros, np.array([1]), np.array([0, 0, 1]), zeros, np.array([1]) ))
            print("👻 [UPPP] ",[round(x, 4) for x in action])
            step_up += 1
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if step_up > 20: done = True
        env.render()
    env.close()

    print(f"\n🎉 Episode {i + 1} finished with total reward: {total_reward}")