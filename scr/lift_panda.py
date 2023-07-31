from math import pi
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2mat, mat2euler

# load default controller parameters for Operational Space Control (OSC)
controller_config = load_controller_config(default_controller="OSC_POSE")

# create an environment to visualize on-screen
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    control_freq=50,
    horizon = 10000,
)
fac_tran = 1
fac_rot = 1
zeros = np.zeros(3)
def move(delta_pos, grip):
    action = np.concatenate((zeros, zeros, np.array([grip]) ))
    done = False
    if abs(delta_pos[0])>0.01 or abs(delta_pos[1])>0.01 or -delta_pos[2]>0.01:
        action = np.concatenate((delta_pos*fac_tran, zeros, np.array([grip]) ))
    else: done = True
    return action, done

def roll(delta_euler, grip):
    delta_euler[:2] = (delta_euler[:2] + pi) % (2 * pi) - pi
    delta_euler[2] = (delta_euler[2] + pi/4) % (2 * pi/4) - pi/4
    action = np.concatenate((zeros, zeros, np.array([grip]) ))
    done = True
    max_idx = np.argmax(np.abs(delta_euler))
    if abs(delta_euler[max_idx]) > 0.02:
        action[max_idx + 3] = delta_euler[max_idx]*fac_rot
        done = False
    # done = False
    # if   abs(delta_euler[0]) > 0.02: action[0+3] = delta_euler[0]*fac_tran
    # elif abs(delta_euler[1]) > 0.02: action[1+3] = delta_euler[1]*fac_tran
    # elif abs(delta_euler[2]) > 0.02: action[2+3] = delta_euler[2]*fac_tran
    # else: done = True
    return action, done

episodes = 1
n = 1
env.reset()
for i in range(episodes):
    env.reset()
    total_reward = 0.
    obs = env._get_observations()
    done = done_roll = done_pick = done_move = done_down = False
    step_pick = step_up = 0
    while not env._check_success():
        # robot_eef_info
        eef_pos = obs['robot0_eef_pos']
        eef_quat = obs['robot0_eef_quat']
        eef_euler = mat2euler(quat2mat(eef_quat))

        # cube_info
        cube_pos = obs['cube_pos']
        cube_quat = obs['cube_quat']
        cube_euler = mat2euler(quat2mat(cube_quat))

        # delta
        delta_pos = cube_pos-eef_pos
        delta_up_pos = np.array([delta_pos[0], delta_pos[1], delta_pos[2]+0.1])
        delta_euler = cube_euler-eef_euler
        delta_euler = np.array([delta_euler[0]-pi, delta_euler[1], delta_euler[2]])

        # action step
        action_roll, done_roll = roll(delta_euler, 0)
        action_move, done_move = move(delta_up_pos, -1)
        action_down, done_down = move(delta_pos, 0)

        if not done_roll:
            action = action_roll
            print("{:03}".format(n), "🤡 [ROLL] ",[round(x/fac_rot, 4) for x in action[:7]],end=" ")
        elif done_roll and not done_move:
            action = action_move
            print("{:03}".format(n), "👾 [MOVE] ",[round(x/fac_tran, 4) for x in action[:7]],end=" ")
        elif done_roll and done_move and not done_down: 
            action = action_down
            print("{:03}".format(n), "🎃 [DOWN] ",[round(x/fac_tran, 4) for x in action[:7]],end=" ")
        elif done_roll and done_move and done_down and not done_pick:
            action = np.concatenate((zeros, zeros, np.array([1])))
            print("{:03}".format(n), "👹 [PICK] ",[round(x, 4) for x in action[:7]],end=" ")
            step_pick += 1
            if step_pick > 20: done_pick = True
        elif done_roll and done_move and done_down and done_pick:
            action =  np.concatenate((np.array([0, 0, 1]), zeros, np.array([1])))
            print("{:03}".format(n), "👻 [UPPP] ",[round(x, 4) for x in action[:7]],end=" ")
            step_up += 1

        obs, reward, done, _ = env.step(action)
        cube_height = env.sim.data.body_xpos[env.cube_body_id][2]
        table_height = env.model.mujoco_arena.table_offset[2]
        print(round(reward, 4), round(cube_height-table_height, 4))
        total_reward += reward
        n += 1
        # if step_up > 20: done = True
        env.render()
    env.close()

    print(f"\n🎉 Episode {i + 1} finished with total reward: {total_reward}")
