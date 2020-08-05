import numpy as np
from scipy.stats import multivariate_normal
import torch
import math
# grid is 6x6, reacher is like 0.4x0.4 but centered at (0,0)

def expert_density(task_name, env, goal=None, goal_radius=None, **kwargs):
    '''
    Generate the state marginal distribution of expert by specifying the reward
    Can be n-modal
    '''
    eps = 1e-8

    # below are robust uniform distributions for any state
    def expert_uniform(state):
        area = env.size_x * env.size_y
        return np.ones((state.shape[0]), dtype=np.float)/area

    def expert_uniform_reacher(state):
        area = math.pi * env.radius**2
        return np.ones((state.shape[0]), dtype=np.float)/area

    def expert_uniform_sawyer(state):
        x_low, y_low = env.puck_goal_low[0], env.puck_goal_low[1]
        x_high, y_high = env.puck_goal_high[0], env.puck_goal_high[1]
        area = (x_high - x_low) * (y_high - y_low)
        return np.ones((state.shape[0]), dtype=np.float) / area

    # def expert_goal(state):
    #     r = goal_radius
    #     x, y = goal
    #     inside = np.logical_and(np.abs(state[:, 0] - x) <= r, np.abs(state[:, 1] - y) <= r)
    #     return inside.astype(np.float) / ((2*r) ** 2) + (1-inside).astype(np.float) * eps
    def expert_uniform_goal(state):
        # a circle
        area = math.pi * goal_radius**2
        inside = np.linalg.norm(state - goal, axis=1) <= goal_radius
        return inside.astype(np.float) / area + (1-inside).astype(np.float) * eps
    
    # def expert_multigoal(state):
    #     all_size = np.sum([(2*r) ** 2 for r in goal_radius])
    #     num = len(goal)
    #     inside = np.zeros(len(state))
    #     for i in range(num):
    #         x, y = goal[i]
    #         r = goal_radius[i]
    #         inside = np.logical_or(inside, np.logical_and(np.abs(state[:, 0] - x) <= r, np.abs(state[:, 1] - y) <= r))
    #     return inside.astype(np.float) / all_size + (1-inside).astype(np.float) * eps

    # def expert_path(state):
    #     size = 7
    #     path = np.logical_or(
    #         np.logical_and(state[:, 0] >= 0, state[:, 0] <= 1),
    #         np.logical_and(state[:, 1] <= 4, state[:, 1] >= 3)
    #     )
    #     return path.astype(np.float) / size  + (1-path).astype(np.float) * eps

    def expert_gaussian(state):
        if isinstance(goal_radius, float):
            r = goal_radius # one std
        else:
            r = np.array(goal_radius) # diagonal std
        return multivariate_normal.pdf(state, mean=goal, cov=r**2)

    def expert_mix_gaussian(state):
        prob = 0.0
        for g, r in zip(goal, goal_radius):
            prob += multivariate_normal.pdf(state, mean=g, cov=r**2)
        return prob / len(goal) # GMM with equal weight
    
    if task_name == 'gaussian':
        return expert_gaussian
    elif task_name == 'mix_gaussian':
        return expert_mix_gaussian    
    elif task_name == 'uniform':
        return expert_uniform
    elif task_name == 'uniform_reacher':
        return expert_uniform_reacher
    elif task_name == 'uniform_sawyer':
        return expert_uniform_sawyer

    elif task_name == 'uniform_goal':
        return expert_uniform_goal

