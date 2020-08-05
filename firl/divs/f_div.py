import numpy as np
import torch
import torch.nn.functional as F

def f_div_loss(div: str, IS: bool, samples, rho_expert, agent_density, reward_func, device):
    # please add eps to expert density, not here
    assert div in ['fkl', 'rkl', 'js']
    s, _, log_a = samples
    N, T, d = s.shape

    s_vec = s.reshape(-1, d)
    log_density_ratio = np.log(rho_expert(s_vec)) - agent_density.score_samples(s_vec).reshape(-1)
    log_density_ratio = torch.FloatTensor(log_density_ratio).to(device)

    if div == 'fkl':
        t1 = torch.exp(log_density_ratio) # (N*T,) p/q TODO: clip
    elif div == 'rkl':
        t1 = log_density_ratio  # (N*T,) log (p/q)
    elif div == 'js':
        t1 = F.softplus(log_density_ratio) # (N*T,) log (1 + p/q)

    t1 = (-t1).view(N, T).sum(1) # NOTE: sign (N,)
    t2 = reward_func.r(torch.FloatTensor(s_vec).to(device)).view(N, T).sum(1) # (N,)    

    if IS:
        traj_reward = reward_func.get_scalar_reward(s_vec).reshape(N, T).sum(1) # (N,)
        traj_log_prob = log_a.sum(1) # (N,)
        IS_ratio = F.softmax(torch.FloatTensor(traj_reward - traj_log_prob), dim=0).to(device) # normalized weight
        surrogate_objective = (IS_ratio * t1 * t2).sum() - (IS_ratio * t1).sum() * (IS_ratio * t2).sum()
    else:
        surrogate_objective = (t1 * t2).mean() - t1.mean() * t2.mean() # sample covariance
    
    surrogate_objective /= T
    return surrogate_objective, t1 / T # log of geometric mean w.r.t. traj (0 is the borderline)

def maxentirl_loss(div: str, agent_samples, expert_samples, reward_func, device):
    ''' NOTE: only for maxentirl (FKL in trajectory): E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    assert div in ['maxentirl']
    sA, _, _ = agent_samples
    _, T, d = sA.shape

    sA_vec = torch.FloatTensor(sA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(expert_samples).reshape(-1, d).to(device)

    t1 = reward_func.r(sA_vec).view(-1) # E_q[r(tau)]
    t2 = reward_func.r(sE_vec).view(-1) # E_p[r(tau)]

    surrogate_objective = t1.mean() - t2.mean() # gradient ascent
    return T * surrogate_objective # same scale

def f_div_current_state_loss(div: str, samples, rho_expert, agent_density, reward_func, device):
    ''' NOTE: deprecated
        div=fkl-state is exactly maxentirl with importance sampling
        div=rkl-state,js-state are approximate SMM-IRL (no theory support)
    '''
    assert div in ['maxentirl', 'fkl-state', 'rkl-state', 'js-state']
    s, _, _ = samples
    N, T, d = s.shape

    s_vec = s.reshape(-1, d)
    log_density_ratio = np.log(rho_expert(s_vec)) - agent_density.score_samples(s_vec).reshape(-1)
    log_density_ratio = torch.FloatTensor(log_density_ratio).to(device)

    if div in ['maxentirl', 'fkl-state']:
        t1 = torch.exp(log_density_ratio) # (N*T,) p/q TODO: clip
    elif div == 'rkl-state':
        t1 = log_density_ratio  # (N*T,) log (p/q)
    elif div == 'js-state':
        t1 = F.softplus(log_density_ratio) # (N*T,) log (1 + p/q)

    t1 = -t1 # NOTE: sign (N*T,)
    t2 = reward_func.r(torch.FloatTensor(s_vec).to(device)).view(-1) # (N*T,) not sum

    surrogate_objective = (t1 * t2).mean() - t1.mean() * t2.mean()
    return T * surrogate_objective # same scale
