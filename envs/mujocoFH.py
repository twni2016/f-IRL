# Fixed Horizon wrapper of mujoco environments
import gym
import numpy as np

class MujocoFH(gym.Env):
    def __init__(self, env_name, T=1000, r=None, obs_mean=None, obs_std=None, seed=1):
        self.env = gym.make(env_name)
        self.T = T
        self.r = r
        assert (obs_mean is None and obs_std is None) or (obs_mean is not None and obs_std is not None)
        self.obs_mean, self.obs_std = obs_mean, obs_std

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.seed(seed)
        
    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        self.t = 0
        self.terminated = False
        self.terminal_state = None

        self.obs = self.env.reset()
        self.obs = self.normalize_obs(self.obs)
        return self.obs.copy()
    
    def step(self, action):
        self.t += 1

        if self.terminated:
            return self.terminal_state, 0, self.t == self.T, True
        else:
            prev_obs = self.obs.copy()
            self.obs, r, done, info = self.env.step(action)
            self.obs = self.normalize_obs(self.obs)
            
            if self.r is not None:  # from irl model
                r = self.r(prev_obs)

            if done:
                self.terminated = True
                self.terminal_state = self.obs
            
            return self.obs.copy(), r, done, done
    
    def normalize_obs(self, obs):
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs
