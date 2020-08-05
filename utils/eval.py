from common.sac import ReplayBuffer, SAC
from utils import system, collect, logger
from utils.plots import train_plot
import torch
from sklearn import neighbors
import numpy as np
import gym
import time, copy

def KL_summary(expert_samples, agent_emp_states, env_steps: int, policy_type: str, show_ent=False):
    start = time.time()
    fkl = collect.forward_kl_knn_based(expert_samples.copy(), agent_emp_states.copy())
    rkl = collect.reverse_kl_knn_based(expert_samples.copy(), agent_emp_states.copy())

    print("*****************************************")
    print(f'env_steps: {env_steps:d}: {policy_type} fkl: {fkl:.3f} rkl: {rkl:.3f} time: {time.time()-start:.0f}s')
    print("*****************************************")

    logger.record_tabular(f"{policy_type} Forward KL", round(fkl, 4))
    logger.record_tabular(f"{policy_type} Reverse KL", round(rkl, 4))

    if show_ent:
        ent = collect.entropy(agent_emp_states)
        print(f'ent: {ent:.3f}')
        logger.record_tabular(f"{policy_type} Entropy", round(ent, 4))
        return {'fkl': fkl, 'rkl': rkl, 'ent': ent}
    else:
        return {'fkl': fkl, 'rkl': rkl}

def evaluate_real_return(policy, env, n_episodes, horizon, deterministic):
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        ret = 0
        for t in range(horizon):
            action = policy(obs, deterministic)
            obs, rew, done, _ = env.step(action) # NOTE: assume rew=0 after done=True for evaluation
            ret += rew 
            if done:
                break
        returns.append(ret)

    return np.mean(returns)

def do_eval(v, reward_func, device):
    '''
    build a new sac, traing it on the current reward till convergence, and then measure the kl divergence / entropy
    '''
    env_name = v['env']['env_name']
    add_time, state_indices = v['env']['add_time'], v['env']['state_indices']
    seed = v['seed']

    eval_env_fn = lambda:gym.make(env_name, r=reward_func, **v['env']) # r for testing sac
    gym_env = eval_env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    eval_replay_buffer = ReplayBuffer(
        state_size, 
        action_size,
        device=device,
        size=v['sac']['buffer_size'])
    
    eval_sac_kwargs = copy.deepcopy(v['sac'])
    eval_sac_kwargs['epochs'] = v['evaluation']['epochs']
    eval_sac_kwargs['random_explore_episodes'] = v['evaluation']['random_explore_episodes']
        
    eval_sac_agent = SAC(eval_env_fn, eval_replay_buffer,
        add_time=add_time,
        update_after=gym_env.T * eval_sac_kwargs['random_explore_episodes'], 
        max_ep_len=gym_env.T,
        seed=seed,
        start_steps=gym_env.T * eval_sac_kwargs['random_explore_episodes'],
        reward_state_indices=state_indices,
        device=device,
        **eval_sac_kwargs
    )
    eval_sac_agent.reward_function = reward_func

    if env_name == "ContinuousVecGridEnv-v0":
        learn_fn, collect_fn = eval_sac_agent.learn, collect.collect_trajectories_policy
    elif env_name in ["ReacherDraw-v0"]:
        learn_fn, collect_fn = eval_sac_agent.learn_mujoco, collect.collect_trajectories_policy_single
    sac_info = learn_fn(n_parallel=1)
    eval_samples = collect_fn(gym_env, eval_sac_agent, n = v['irl']['training_trajs'], state_indices=state_indices)

    return sac_info, eval_samples[0]

def do_eval_reuse(v, eval_sac_agent, reward_func):
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']

    if env_name == "ContinuousVecGridEnv-v0":
        learn_fn, collect_fn = eval_sac_agent.learn, collect.collect_trajectories_policy
        eval_sac_agent.test_env = gym.make(env_name, r=reward_func, **v['env'])
    elif env_name in ["ReacherDraw-v0"]:
        learn_fn, collect_fn = eval_sac_agent.learn_mujoco, collect.collect_trajectories_policy_single
        eval_sac_agent.test_env = gym.make(env_name, **v['env'])
        
    train_env = gym.make(env_name, **v['env'])  # not need r since batch['rew']
    gym_env = gym.make(env_name, **v['env'])
    # print(eval_sac_agent.replay_buffer.size)
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    eval_sac_agent.reinitialize = False
    eval_sac_agent.env = train_env
    eval_sac_agent.epochs = v['evaluation']['epochs']
    eval_sac_agent.reward_function = reward_func # important

    # below are same as do_eval()
    sac_info = learn_fn(print_out=True, n_parallel=1)
    eval_samples = collect_fn(gym_env, eval_sac_agent, n = v['irl']['training_trajs'], state_indices=state_indices)

    return sac_info, eval_samples[0]
