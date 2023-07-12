from math import pi
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler

controller_config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    env_name="MaholoLaboratory",
    robots="Maholo",
    gripper_types=["PandaGripper", "PandaGripper"],
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    control_freq=50,
    horizon = 2000,
)

fac_tran = 1
fac_rot = 1
zeros = np.zeros(3)
def move(delta_pos, grip):
    action = np.concatenate((zeros, zeros, np.array([grip]) ))
    done = False
    if abs(delta_pos[0])>0.05 or abs(delta_pos[1])>0.05 or -delta_pos[2]>0.05:
        action = np.concatenate((delta_pos*fac_tran, zeros, np.array([grip]) ))
    else: done = True
    return action, done

def roll(delta_euler, grip):
    # delta_euler[0] = (delta_euler[0] + pi) % (2 * pi) - pi
    # delta_euler[2] = (delta_euler[2] + pi/2) % (2 * pi/2) - pi/2
    action = np.concatenate(( zeros, zeros, np.array([grip]) ))
    # done = False
    # if   abs(delta_euler[0]) > 0.02: action[0+3] = delta_euler[0]*fac_rot
    # elif abs(delta_euler[1]) > 0.02: action[1+3] = delta_euler[1]*fac_rot
    # elif abs(delta_euler[2]) > 0.02: action[2+3] = delta_euler[2]*fac_rot
    # else: done = True
    done = True
    max_idx = np.argmax(np.abs(delta_euler))
    if abs(delta_euler[max_idx]) > 0.05:
        action[max_idx + 3] = delta_euler[max_idx]*fac_rot
        done = False
    return action, done

done = done_left_roll = done_pick = done_left_pos_preto_pipette = done_left_pos_to_pipette = done_left_pos_preto_tube = done_left_pos_to_tube = False
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

        pipette_quat = obs['pipette_quat']
        pipette_euler = mat2euler(quat2mat(pipette_quat))
        pipette_pos = obs["pipette_pos"]
        tube_pos = obs["tube_pos"]

        # delta
        delta_left_pos_to_pipette = pipette_pos - eef_left_pos
        delta_left_pos_to_pipette[2] += 0.5
        delta_left_pos_preto_pipette = delta_left_pos_to_pipette
        delta_left_pos_preto_pipette[1] += 0.2

        delta_left_pos_to_tube = tube_pos - pipette_pos
        delta_left_pos_to_tube[2] += 0.2
        delta_left_pos_preto_tube = delta_left_pos_to_tube
        delta_left_pos_preto_tube[2] += 0.2

        delta_left_euler = pipette_euler - eef_left_euler
        delta_left_euler[1] += pi/2

        # action step
        action_right_roll = np.zeros(7)
        action_left_roll, done_left_roll = roll(delta_left_euler, -1)

        action_right_move = np.zeros(7)
        action_left_pos_preto_pipette, done_left_pos_preto_pipette = move(delta_left_pos_preto_pipette, -1)
        action_left_pos_to_pipette,    done_left_pos_to_pipette    = move(delta_left_pos_to_pipette,    -1)
        action_left_pos_preto_tube, done_left_pos_preto_tube = move(delta_left_pos_preto_tube, -1)
        action_left_pos_to_tube,    done_left_pos_to_tube    = move(delta_left_pos_to_tube,    -1)
        
        # action = np.concatenate((action_right_roll, action_left_roll))
        # print("🤡 [ROLL] ",[round(x/fac_rot, 4) for x in action])
        if not done_left_pos_preto_pipette and not done_left_roll:
            if not done_left_pos_preto_pipette:
                action = np.concatenate((action_right_move, action_left_pos_preto_pipette))
                print("👾 [MOVE] ",[round(x/fac_tran, 4) for x in action])
            if not done_left_roll:
                action = np.concatenate((action_right_roll, action_left_roll))
                print("🤡 [ROLL] ",[round(x/fac_rot, 4) for x in action])
        elif not done_left_pos_to_pipette:
            action = np.concatenate((action_right_move, action_left_pos_to_pipette))
            print("🎃 [MOVE] ",[round(x/fac_tran, 4) for x in action])
        elif not done_pick:
            action = np.concatenate((zeros, zeros, np.array([1]), zeros, zeros, np.array([1]) ))
            print("👹 [PICK] ",[round(x, 4) for x in action])
            step_pick += 1
            if step_pick > 30: done_pick = True
        elif not done_left_pos_preto_tube:
            action = np.concatenate((action_right_move, action_left_pos_preto_tube))
            print("👻 [MOVE] ",[round(x/fac_tran, 4) for x in action])
        elif not done_left_pos_to_tube:
            action = np.concatenate((action_right_move, action_left_pos_to_tube))
            print("👽 [MOVE] ",[round(x/fac_tran, 4) for x in action])
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
    env.close()

    print(f"\n🎉 Episode {i + 1} finished with total reward: {total_reward}")