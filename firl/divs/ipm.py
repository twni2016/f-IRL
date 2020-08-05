import numpy as np
import torch
import torch.nn.functional as F

def ipm_loss(metric: str, IS: bool, samples, critic_value, reward_func, device, expert_trajs=None):
    # please add eps to expert density, not here
    assert metric in ['emd']
    s, _, log_a = samples
    if expert_trajs is not None:
        assert expert_trajs.ndim == 3 and expert_trajs.shape[1:] == s.shape[1:]
        s = np.concatenate((s, expert_trajs), axis=0)
    N, T, d = s.shape

    s_vec = s.reshape(-1, d)
    logits = critic_value(s_vec) # torch vector

    t1 = (-logits).view(N, T).sum(1) # NOTE: sign (N,)
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
