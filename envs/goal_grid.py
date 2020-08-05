import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np

class GoalContinuousGrid(gym.Env):
    '''
    doc
    '''

    def __init__(self, r=None, size_x=6, size_y=6, T=30, prior_reward_weight=0.1, random_born=False, state_indices=[0,1],
        random_act_prob=0.0, sigma=1.0, terminal_states=[], seed=0, add_time=False, **kwargs):
        self.size_x = size_x
        self.size_y = size_y
        self.terminal_states = terminal_states
        self.r = r
        self.range_x = (0, size_x)
        self.range_y = (0, size_y)
        self.random_act_prob = random_act_prob
        self.sigma = sigma
        self.add_time = add_time
        self.prior_reward_weight = prior_reward_weight
        self.mode = 'train'

        self.T = T
        if not add_time:
            self.observation_space = Box(low=np.array([0,0]),high=np.array([size_x,size_y]),dtype=np.float32)
        else:
            self.observation_space = Box(low=np.array([0,0,1]),high=np.array([size_x,size_y,T]),dtype=np.float32)

        self.action_space = Box(low=np.array([-1,-1]),high=np.array([1,1]),dtype=np.float32)

        self.seed(seed)
        self.action_space.seed(seed)
        self.random_born = random_born

        if self.r is not None:
            n = 100
            x = np.linspace(0, self.size_x, n)
            y = np.linspace(0, self.size_y, n)
            xx, yy = np.meshgrid(x, y)
            zz = np.stack([xx.flatten(), yy.flatten()], axis=1)
            all_reward = self.r(zz)
            self.min_prior_reward, self.max_prior_reward = np.min(all_reward), np.max(all_reward)
            self.prior_reward_range = self.max_prior_reward - self.min_prior_reward


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
        if not self.add_time:
            return self.s.copy()
        else:
            ret = np.zeros((n, 3))
            ret[:, :2] = self.s.copy()
            ret[:, 2] = self.t
            return ret

    def step(self, action):
        change_action_prob = (np.random.uniform(0, 1, size=(self.n)) < self.random_act_prob).reshape(-1,1)
        action = change_action_prob * (action + self.sigma * np.random.randn(self.n, 2)) \
                + (1-change_action_prob) * action

        old_s = self.s.copy()

        self.s += action   
        self.s[:,0] = np.clip(self.s[:,0],0,self.size_x)
        self.s[:,1] = np.clip(self.s[:,1],0,self.size_y)

        new_s = self.s.copy()

        self.t += 1
        done = (self.t >= self.T) 

        reward = np.logical_and(self.s[:, 0] > self.size_x - 0.05, self.s[:, 1] > self.size_y - 0.05).astype(np.float32)
        reward_distractor = np.logical_and(self.s[:, 0] > self.size_x - 0.05, self.s[:, 1] < 0.05).astype(np.float32) * 0.1
        reward_distractor2 = np.logical_and(self.s[:, 1] > self.size_y - 0.05, self.s[:, 0] < 0.05).astype(np.float32) * 0.1
        reward += reward_distractor
        reward += reward_distractor2

        if self.r is not None and self.mode == 'train':
            # prior_reward = self.r(self.s)
            # prior_reward = (prior_reward - self.min_prior_reward) / self.prior_reward_range
            # reward += self.prior_reward_weight * prior_reward

            prior_reward = self.r(new_s) - self.r(old_s)
            reward += self.prior_reward_weight * prior_reward

        if not self.add_time:
            return self.s.copy(), reward, done, None
        else:
            ret = np.zeros((self.s.shape[0], 3))
            ret[:, :2] = self.s.copy()
            ret[:, 2] = self.t
            return ret, reward, done, None

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'
