import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os 
# NOTE: range of finertip is within 0.21 radius (except 0.01 radius),
# the arena length is [-0.3, -0.3] x [0.3, 0.3], z=0.01 
# The object/target is removed from xml

class ReacherTraceEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, add_time=False, T=100, r=None, state_indices=[0,1], **kwargs):

        self.T = T
        self.add_time = add_time
        self.state_indices = state_indices
        self.r = r
        self.range_x = (-0.21, 0.21)
        self.range_y = (-0.21, 0.21)
        self.radius = 0.21

        utils.EzPickle.__init__(self)
        full_path = os.path.join(os.path.dirname(__file__), "assets", 'reacher_no_target.xml')
        mujoco_env.MujocoEnv.__init__(self, full_path, 2)

    def step(self, a):

        # vec = self.get_body_com("fingertip")-self.get_body_com("target")
        # reward_dist = - np.linalg.norm(vec)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        if self.r is None:
            reward = 0.0
        else:
            reward = self.r(ob[None, self.state_indices])

        done = False # forever False
        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        
    def reset(self):
        obs = super().reset()
        return obs

    def reset_model(self):
        # init_qpos: (0, 0), init fingertip at (0.21, 0), and sample rad from U(-0.1, 0.1) 
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # while True:
        #     self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        #     if np.linalg.norm(self.goal) < 0.2:
        #         break
        qpos[-2:] = 0 # self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        obs = np.concatenate([
            self.get_body_com("fingertip"), # 3-dim fixed z=0.01
            np.cos(theta),
            np.sin(theta),
            # self.sim.data.qpos.flat[2:], # 2-dim target
            self.sim.data.qvel.flat[:2], # vx, vy
        ])

        return obs.copy()