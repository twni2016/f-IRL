import numpy as np
import torch
import torch.nn.functional as F

def f_div_disc_loss(div: str, IS: bool, samples, disc, reward_func, device, expert_trajs=None):
    # please add eps to expert density, not here
    assert div in ['fkl', 'rkl', 'js']
    s, _, log_a = samples
    if expert_trajs is not None:
        assert expert_trajs.ndim == 3 and expert_trajs.shape[1:] == s.shape[1:]
        s = np.concatenate((s, expert_trajs), axis=0) # NOTE: this won't change samples variable
    N, T, d = s.shape

    s_vec = s.reshape(-1, d)
    logits = disc.log_density_ratio(s_vec) # torch vector

    if div == 'fkl':
        t1 = torch.exp(logits) # (N*T,) p/q TODO: clip
    elif div == 'rkl':
        t1 = logits # (N*T,) log (p/q)
    elif div == 'js':  # https://pytorch.org/docs/master/generated/torch.nn.Softplus.html
        t1 = F.softplus(logits) # (N*T,) log (1 + p/q)

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

