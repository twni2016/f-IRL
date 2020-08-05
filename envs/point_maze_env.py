import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

# from rllab.misc import logger

from envs.dynamic_mjc.mjc_models import point_mass_maze

# start from [length/2,0], target at [length/2,length-0.1]
# two spheres with size 0.03 and friction
# border at (-0.1, -0.1) x (length, length)
# block at (-0.1, 2/3 length) or (1/3 length, length) at length/2 with size 0.02 with capsule shape
class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, direction=1, maze_length=0.6, r=None, sparse_reward=False, no_reward=False, T=100, **kwargs):
        utils.EzPickle.__init__(self)
        self.r = r
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = T
        self.direction = direction
        self.length = maze_length

        self.episode_length = 0

        self.size_x = self.size_y = maze_length + 0.1
        self.range_lim = [[-0.1, maze_length], [-0.1, maze_length]]

        model = point_mass_maze(direction=self.direction, length=self.length)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        vec_dist = self.get_body_com("particle") - self.get_body_com("target")

        reward_dist = - np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = - np.square(a).sum()
        if self.r is not None:
            reward = self.r(self._get_obs())
        elif self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if reward_dist <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.get_body_com("particle")[:2], # ignore z constant zero
            #self.get_body_com("target"),
        ])

    def plot_trajs(self, *args, **kwargs):
        pass

    def log_diagnostics(self, paths):
        rew_dist = np.array([traj['env_infos']['reward_dist'] for traj in paths])
        rew_ctrl = np.array([traj['env_infos']['reward_ctrl'] for traj in paths])

        logger.record_tabular('AvgObjectToGoalDist', -np.mean(rew_dist.mean()))
        logger.record_tabular('AvgControlCost', -np.mean(rew_ctrl.mean()))
        logger.record_tabular('AvgMinToGoalDist', np.mean(np.min(-rew_dist, axis=1)))

