import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

class ContinuousGridEnv(gym.Env):

    def __init__(self, r=None, size_x=4, size_y=4, T=50, random_born=False, state_indices=None,
        random_act_prob=0.0, sigma=1.0, terminal_states=[], seed=0, add_time=False, **kwargs):
        self.size_x = size_x
        self.size_y = size_y
        self.terminal_states = terminal_states
        self.r = r
        self.range_x = (0, size_x)
        self.range_y = (0, size_y)
        self.random_act_prob = random_act_prob
        self.sigma = sigma
        self.state_indices = state_indices
        self.T = T

        self.observation_space = Box(low=np.array([0,0]),high=np.array([size_x,size_y]),dtype=np.float32)
        self.action_space = Box(low=np.array([-1,-1]),high=np.array([1,1]),dtype=np.float32)

        self.seed(seed)
        self.action_space.seed(seed)
        self.random_born = random_born

    def set_reward_function(self,r):
        self.r = r

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, n=1):
        if self.random_born:
            self.s = np.random.uniform((0,0),(self.size_x,self.size_y),size=(n, 2))
        else:
            self.s = np.zeros((n, 2), dtype=np.float32)

        self.n = n
        self.t = 0
        return self.s.copy()

    def step(self, action):
        change_action_prob = (np.random.uniform(0, 1, size=(self.n)) < self.random_act_prob).reshape(-1,1)
        action = change_action_prob * (action + self.sigma * np.random.randn(self.n, 2)) \
                + (1-change_action_prob) * action
        self.s += action   
        self.s[:,0] = np.clip(self.s[:,0],0,self.size_x)
        self.s[:,1] = np.clip(self.s[:,1],0,self.size_y)
        self.t += 1
        done = (self.t >= self.T) 
        if self.r is None: # for adv IRL
            r = np.zeros((self.n,))
        else:              # for SMM IRL
            r= self.r(self.s)

        return self.s.copy(), r, done, None
