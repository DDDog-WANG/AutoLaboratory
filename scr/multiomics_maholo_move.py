import robosuite as suite
import numpy as np

def maholo_move(obs):

    delta_pos = obs["g0_to_target_pos"]

    zeros = np.zeros(3)
    action_right = np.concatenate((delta_pos*5, zeros, np.array([1])))
    action_left = np.concatenate((zeros, zeros, np.array([1])))
    action = np.concatenate((action_right, action_left))

    return action








