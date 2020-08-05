import sys, os, time
from ruamel.yaml import YAML
from utils import system

import gym
import numpy as np 
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import envs
from common.sac import ReplayBuffer, SAC
from utils.plots.train_plot import plot_sac_curve

def train_policy(EnvCls):
    env = EnvCls()
    env.seed(seed)
    
    replay_buffer = ReplayBuffer(
        env.observation_space.shape[0], 
        env.action_space.shape[0],
        device=device,
        size=v['sac']['buffer_size'])
    
    sac_agent = SAC(EnvCls, replay_buffer,
        steps_per_epoch=env_T,
        update_after=env_T * v['sac']['random_explore_episodes'], 
        max_ep_len=env_T,
        seed=seed,
        start_steps=env_T * v['sac']['random_explore_episodes'],
        device=device,
        **v['sac']
        )
    assert sac_agent.reinitialize == True

    sac_agent.test_fn = sac_agent.test_agent_ori_env
    sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps = sac_agent.learn_mujoco(print_out=True)

    plot_sac_curve(axs[0], sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps)

    return sac_agent.get_action

def evaluate_policy(policy, env, n_episodes, deterministic=False):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    expert_states = torch.zeros((n_episodes, env_T, obs_dim)) # s1 to sT
    expert_actions = torch.zeros((n_episodes, env_T, act_dim)) # a0 to aT-1

    returns = []

    for n in range(n_episodes):
        obs = env.reset()
        ret = 0
        for t in range(env_T):
            action = policy(obs, deterministic)
            obs, rew, _, _ = env.step(action) # NOTE: assume rew=0 after done=True for evaluation
            expert_states[n, t, :] = torch.from_numpy(obs).clone()
            expert_actions[n, t, :] = torch.from_numpy(action).clone()
            ret += rew
        returns.append(ret)
    
    return expert_states, expert_actions, np.array(returns)


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name, env_T = v['env']['env_name'], v['env']['T']
    seed = v['seed']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    EnvCls = lambda : gym.make(env_name, T=env_T)
    print(f"training Expert on {env_name}")
    policy = train_policy(EnvCls)

    env = EnvCls()
    env.seed(seed+1)

    expert_states_sto, expert_actions_sto, expert_returns = evaluate_policy(policy, env, v['expert']['samples_episode'])
    return_info = f'Expert(Sto) Return Avg: {expert_returns.mean():.2f}, std: {expert_returns.std():.2f}'
    print(return_info)

    log_txt = open(f"expert_data/meta/{env_name}_{seed}.txt", 'w')
    log_txt.write(return_info + '\n')
    log_txt.write(repr(expert_returns)+'\n')

    sns.violinplot(data=expert_returns, ax=axs[1])
    axs[1].set_title("violin plot of expert return")
    plt.savefig(os.path.join(f'expert_data/meta/{env_name}_{seed}.png')) 

    expert_states_det, expert_actions_det, expert_returns = evaluate_policy(policy, env, v['expert']['samples_episode'], True)
    return_info = f'Expert(Det) Return Avg: {expert_returns.mean():.2f}, std: {expert_returns.std():.2f}'
    print(return_info)
    log_txt.write(return_info + '\n')
    log_txt.write(repr(expert_returns)+'\n')

    log_txt.write(repr(v))

    os.makedirs('expert_data/states/', exist_ok=True)
    os.makedirs('expert_data/actions/', exist_ok=True)
    torch.save(expert_states_sto, f'expert_data/states/{env_name}_{seed}_sto.pt')
    torch.save(expert_states_det, f'expert_data/states/{env_name}_{seed}_det.pt')
    torch.save(expert_actions_sto, f'expert_data/actions/{env_name}_{seed}_sto.pt')
    torch.save(expert_actions_det, f'expert_data/actions/{env_name}_{seed}_det.pt')