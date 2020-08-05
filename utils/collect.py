import numpy as np
import torch
from utils.it_estimator import entropy as it_entropy
from utils.it_estimator import kldiv
from scipy.stats import multivariate_normal

# Collect samples using the SAC policy
def collect_trajectories_policy(env, sac_agent, n=10000, state_indices=None):
    '''
    Samples n trajectories from env using sac_agent 
    
    :return: N trajectory samples
            Tuple of NxTx|S| state array, Nx(T-1) action array, Nx(T-1) action probs array
            # Nx(T-1) reward array
    '''
    T = env.T
    s_buffer = np.empty((n, T, env.observation_space.shape[0]), dtype=np.float32)
    a_buffer = np.empty((n, T-1, env.action_space.shape[0]), dtype=np.float32)
    log_a_buffer = np.empty((n, T-1))
    # r_buffer = np.empty((n, T-1))

    s = env.reset(n)
    for i in range(T-1):
        a, logpi = sac_agent.get_action_batch(s)

        s_nxt, _, _, _ = env.step(a) # assign reward online

        s_buffer[:,i,:] = s
        a_buffer[:,i,:] = a
        # r_buffer[:,i] = r
        log_a_buffer[:,i] = logpi
        s = s_nxt

    s_buffer[:, T-1, :] = s
    s_buffer = s_buffer[:,1:,:]   # NOTE

    if state_indices is None:
        return s_buffer, a_buffer, log_a_buffer
    else:
        return s_buffer[:, :, state_indices], a_buffer, log_a_buffer



def collect_trajectories_policy_single(env, sac_agent, n=2000, state_indices=None, render=False):
    T = sac_agent.max_ep_len
    s_buffer = np.empty((n, T+1, env.observation_space.shape[0]), dtype=np.float32)
    a_buffer = np.empty((n, T, env.action_space.shape[0]), dtype=np.float32)
    log_a_buffer = np.empty((n, T))


    for traj_no in range(n):

        s = env.reset()
        for i in range(T):
            a, logpi = sac_agent.get_action(s,get_logprob=True)
            s_nxt, _, _, _ = env.step(a) # assign reward online
            s_buffer[traj_no,i,:] = s
            a_buffer[traj_no,i,:] = a
            log_a_buffer[traj_no,i] = logpi
            s = s_nxt
            if render:
                env.render()
        s_buffer[traj_no, T, :] = s

    s_buffer = s_buffer[:,1:,:]   # NOTE

    if state_indices is None:
        return s_buffer, a_buffer, log_a_buffer
    else:
        return s_buffer[:, :, state_indices], a_buffer, log_a_buffer


# for KL evaluation
def rejection_sampling(rho_expert, task, env, n=1000, goal_radius=0.5):
    # proposal: uniform distribution on grid 
    # k = max (P(x) / Q(x))
    assert task == 'uniform'
    k = 2 # 4.0/math.pi

    size_x, size_y = env.size_x, env.size_y
    Q_density = 1.0/(size_x * size_y)
    Q_samples = np.random.uniform((0,0),(size_x,size_y),size=(n, 2))

    accepts = np.random.uniform(0, 1, size=(n)) <= (rho_expert(Q_samples) / (k * Q_density)) # u <= p(x) / (k * q(x))
    return Q_samples[accepts]


# credit to http://joschu.net/blog/kl-approx.html
# use unbiased, low-variance, nonnegative estimator by John Schulman: f(r) - f'(1) * (r - 1) >= 0
# intuition: E_q[r] = 1, negatively correlated

# E_q [log q/p] = E_q [(r - 1) - log r], r = p/q
def reverse_kl_density_based(agent_states, rho_expert, agent_density):
    r = np.clip(rho_expert(agent_states), a_min=1e-8, a_max=None) / np.exp(agent_density.score_samples(agent_states))
    return np.mean(r - 1 - np.log(r))

# E_p [log p/q] = E_p [(r - 1) - log r], r = q/p
def forward_kl_density_based(expert_states, rho_expert, agent_density):
    r = np.clip(np.exp(agent_density.score_samples(expert_states)), a_min=1e-8, a_max=None) / rho_expert(expert_states)
    return np.mean(r - 1 - np.log(r))

# NOTE: the above KL estimator is inaccurate especially for disjoint distributions. But they are smooth to plot and compare.
# If we want accurate estimator, please use it_estimator.kldiv() as below

def reverse_kl_knn_based(expert_states, agent_states):
    return kldiv(agent_states, expert_states)

def forward_kl_knn_based(expert_states, agent_states):
    return kldiv(expert_states, agent_states)

def entropy(agent_states):
    return it_entropy(agent_states)

def expert_samples(env_name, task, rho_expert, range_lim):
    trials = 0
    n = task['expert_samples_n']
    while True:
        s = rejection_sampling(env_name, task, rho_expert, range_lim, n)
        if trials == 0:
            samples = s
        else:
            samples = np.concatenate((samples, s), axis=0)
        print(f"trial {trials} samples {samples.shape[0]}")
        trials += 1
        
        if samples.shape[0] >= n:
            return samples

# for KL evaluation
def rejection_sampling(env_name, task, rho_expert, range_lim, n=1000):
    # proposal: uniform distribution
    # k = max (P(x) / Q(x))
    assert 'uniform' in task['task_name']
    k = 2 # 4.0/math.pi

    range_x, range_y = range_lim
    Q_density = 1.0/((range_x[1]-range_x[0]) * (range_y[1]-range_y[0]))
    Q_samples = np.random.uniform((range_x[0],range_y[0]),(range_x[1],range_y[1]),size=(n, 2))
    
    accepts = np.random.uniform(0, 1, size=(n)) <= (rho_expert(Q_samples) / (k * Q_density)) # u <= p(x) / (k * q(x))
    return Q_samples[accepts]

def get_range_lim(env_name, task, env):
    # TODO: change to low, high
    if env_name in ["ContinuousVecGridEnv-v0", "ReacherDraw-v0"]:
        range_x, range_y = env.range_x, env.range_y
    elif env_name == "PointMazeRight-v0":
        return env.range_lim
    return [range_x, range_y]        

def gaussian_samples(env_name, task, env, range_lim):
    range_x, range_y = range_lim
    n = task['expert_samples_n']
    if env_name in ['ContinuousVecGridEnv-v0', "ReacherDraw-v0"]:    
        if task['task_name'] == 'gaussian':
            if isinstance(task['goal_radius'], float):
                r = task['goal_radius']
            else:
                r = np.array(task['goal_radius'])
            samples = multivariate_normal.rvs(mean=task['goal'], cov=r**2, size=n)
        elif task['task_name'] == 'mix_gaussian':
            m = len(task['goal'])
            z = np.random.choice(m, size=n) # assume equal prob
            samples = []
            for g, r in zip(task['goal'], task['goal_radius']):
                samples.append(multivariate_normal.rvs(mean=g, cov=r**2, size=n))
            samples = np.array(samples) # (m, n, 2)
            samples = np.take_along_axis(samples, z[None, :, None], axis=0)[0] # like torch.gather, (n, 2)

        if env_name == "ReacherDraw-v0":
            accepts = (samples[:, 0] ** 2 + samples[:, 1] ** 2) <= env.radius **2
        elif env_name in ["ContinuousVecGridEnv-v0"]:
            x_bool = np.logical_and(samples[:, 0] <= range_x[1], samples[:, 0] >= range_x[0])
            y_bool = np.logical_and(samples[:, 1] <= range_y[1], samples[:, 1] >= range_y[0])
            accepts = np.logical_and(x_bool, y_bool)
        else:
            raise NotImplementedError

        print(f"accepts {accepts.sum()}")
        return samples[accepts] # discard samples outside support does not change KL ordering

