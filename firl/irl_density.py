'''
f-IRL: Extract policy/reward from specified expert density
'''

import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML

from firl.divs.f_div import f_div_loss, f_div_current_state_loss
from firl.divs.ipm import ipm_loss
from firl.models.reward import MLPReward
from firl.models.discrim import SMMIRLDisc as Disc
from firl.models.discrim import SMMIRLCritic as Critic
from common.sac import ReplayBuffer, SAC

import envs
from envs.tasks.grid_task import expert_density
from utils import system, collect, logger, eval
from utils.plots.train_plot import plot, plot_disc, plot_submission
from sklearn import neighbors 

import datetime
import dateutil.tz
import json, copy

def try_evaluate(itr: int, policy_type: str, sac_info, old_reward=None):
    assert policy_type in ["Running"]
    update_time = itr * v['reward']['gradient_step']
    env_steps = itr * v['sac']['epochs'] * v['env']['T']

    agent_emp_states = samples[0].copy()

    metrics = eval.KL_summary(expert_samples, agent_emp_states.reshape(-1, agent_emp_states.shape[2]), 
                         env_steps, policy_type, task_name == 'uniform')

    if v['obj'] in ["emd"]:
        eval_len = int(0.1 * len(critic_loss["main"]))
        emd = -np.array(critic_loss["main"][-eval_len:]).mean()
        metrics['emd'] = emd
        logger.record_tabular(f"{policy_type} EMD", emd)
        plot_disc(agent_emp_states, reward_func.get_scalar_reward, critic.value, v['obj'],
                log_folder, env_steps, range_lim,
                sac_info, critic_loss, metrics)
        
    elif v['density']['model'] == "disc":
        plot_disc(agent_emp_states, reward_func.get_scalar_reward, disc.log_density_ratio, v['obj'],
                log_folder, env_steps, range_lim,
                sac_info, disc_loss, metrics)
    elif env_name == 'ReacherDraw-v0':
        plot_submission(agent_emp_states, reward_func.get_scalar_reward, v['obj'],
                log_folder, env_steps, range_lim, metrics, rho_expert)
    else: # kde
        plot(agent_emp_states, reward_func.get_scalar_reward, agent_density.score_samples,
                lambda x: np.log(rho_expert(x)) - agent_density.score_samples(x), v['obj'],
                log_folder, env_steps, range_lim,
                sac_info, metrics, reward_losses, old_reward=old_reward.get_scalar_reward)

    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name, task_name = v['env']['env_name'], v['task']['task_name']
    add_time, state_indices = v['env']['add_time'], v['env']['state_indices']
    seed = v['seed']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert v['obj'] in ['fkl', 'rkl', 'js', 'emd', 'maxentirl']
    assert task_name in ['uniform', 'gaussian', 'mix_gaussian', 'uniform_goal']
    assert v['density']['model'] in ['kde', 'disc']
    assert add_time == False
    assert v['IS'] == False
    assert env_name in ["ContinuousVecGridEnv-v0", "ReacherDraw-v0"]

    # logs
    exp_id = f"logs/{env_name}/8-4-{task_name}/{v['obj']}" # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)            
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp firl/irl_density.py {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    if v['irl']['save_interval'] > 0:
        os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name, **v['env'])
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]

    # rho_expert and samples for KL estimation (not training)
    rho_expert = expert_density(**v['task'], env=gym_env)
    range_lim = collect.get_range_lim(env_name, v['task'], gym_env)

    if task_name in ['gaussian', 'mix_gaussian']:
        expert_samples = collect.gaussian_samples(env_name, v['task'], gym_env, range_lim)
    elif 'uniform' in task_name:
        expert_samples = collect.expert_samples(env_name, v['task'], rho_expert, range_lim)

    # Initilialize reward as a neural network
    reward_func = MLPReward(len(state_indices), **v['reward'], device=device).to(device)
    reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=v['reward']['lr'], 
        weight_decay=v['reward']['weight_decay'], betas=(v['reward']['momentum'], 0.999))

    for itr in range(v['irl']['n_itrs']):
        # make a new environment with current reward
        if env_name in ["ContinuousVecGridEnv-v0"]:
            env_fn = lambda:gym.make(env_name, r=reward_func.get_scalar_reward, **v['env'])

        if v['sac']['reinitialize'] or itr == 0:
            # Reset SAC agent with old policy, new environment, and new replay buffer
            print("Reinitializing sac")
            replay_buffer = ReplayBuffer(
                state_size, 
                action_size,
                device=device,
                size=v['sac']['buffer_size'])
                
            sac_agent = SAC(env_fn, replay_buffer,
                add_time=add_time,
                steps_per_epoch=v['env']['T'],
                update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
                max_ep_len=v['env']['T'],
                seed=seed,
                start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
                reward_state_indices=state_indices,
                device=device,
                **v['sac']
            )

            if env_name == "ContinuousVecGridEnv-v0":
                learn_fn, collect_fn = sac_agent.learn, collect.collect_trajectories_policy
            else:
                learn_fn, collect_fn = sac_agent.learn_mujoco, collect.collect_trajectories_policy_single
        else:
            if env_name == "ContinuousVecGridEnv-v0":
                sac_agent.env = env_fn()
                sac_agent.test_env = env_fn()
        
        sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac
        sac_info = learn_fn()

        start = time.time()
        samples = collect_fn(gym_env, sac_agent, n = v['irl']['training_trajs'], state_indices=state_indices)
        # Fit a density model using the samples
        agent_emp_states = samples[0].copy()
        agent_emp_states = agent_emp_states.reshape(-1,agent_emp_states.shape[2]) # n*T states
        print(f'collect trajs {time.time() - start:.0f}s', flush=True)

        if v['obj'] in ["emd"]:
            if v['critic']["reinitialize"] or itr == 0:
                critic = Critic(len(state_indices), **v['critic'], device=device)
            start = time.time()
            critic_loss = critic.learn(expert_samples.copy(), agent_emp_states, iter=v['critic']['iter'])
            print(f'train critic {time.time() - start:.0f}s', flush=True)
        # Initialize a density model using KDE
        elif v['density']['model'] == 'kde':
            agent_density = neighbors.KernelDensity(bandwidth=v['density']['kde']['bandwidth'], kernel=v['density']['kde']['kernel'])
            agent_density.fit(agent_emp_states)
        elif v['density']['model'] == "disc":
            start = time.time()
            # learn log density ratio
            disc = Disc(len(state_indices), **v['density']['disc'], device=device)
            disc_loss = disc.learn(expert_samples.copy(), agent_emp_states, iter=v['density']['disc']['iter'])
            print(f'train disc {time.time() - start:.0f}s', flush=True)
       
        old_reward = copy.deepcopy(reward_func)

        # optimization w.r.t. reward
        reward_losses = []
        for _ in range(v['reward']['gradient_step']):
            if v['obj'] in ['fkl', 'rkl', 'js']:
                loss, _ = f_div_loss(v['obj'], v['IS'], samples, rho_expert, agent_density, reward_func, device)            
            elif v['obj'] in ['maxentirl']:
                loss = f_div_current_state_loss(v['obj'], samples, rho_expert, agent_density, reward_func, device)
            elif v['obj'] == 'emd':
                loss, _ = ipm_loss(v['obj'], v['IS'], samples, critic.value, reward_func, device)  
            
            reward_losses.append(loss.item())
            print(f"{v['obj']} loss: {loss}")
            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()
        
        # evaluating the learned reward
        try_evaluate(itr, "Running", sac_info, old_reward)

        logger.record_tabular("Itration", itr)
        logger.record_tabular("Reward Loss", loss.item())
        
        if v['irl']['save_interval'] > 0 and (itr % v['irl']['save_interval'] == 0 or itr == v['irl']['n_itrs']-1):
            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), f"model/reward_model_{itr}.pkl"))

        logger.dump_tabular()