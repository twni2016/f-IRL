import sys, os, time
import numpy as np
import math
import gym
from ruamel.yaml import YAML
import envs

import torch
from common.sac import SAC
from baselines.discrim import ResNetAIRLDisc, MLPDisc
from baselines.adv_smm import AdvSMM

from utils import system, collect, logger
import datetime
import dateutil.tz
import json

if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert v['obj'] in ['f-max-rkl', 'gail', 'fairl', 'airl'] # approximate [RKL, JSD, FKL, RKL]

    # logs
    exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{v['obj']}"  # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)        
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp baselines/main_samples.py {log_folder}')
    os.system(f'cp baselines/adv_smm.py {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    print('pid', os.getpid())
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    os.makedirs(os.path.join(log_folder, 'model'))

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    if v['obj'] != 'airl':
        load_path = f'expert_data/states/{env_name}.pt' 
    else:
        load_path = f'expert_data/states/{env_name}_airl.pt' 
    expert_trajs = torch.load(load_path).numpy()[:, :, state_indices]
    print(num_expert_trajs)
    expert_trajs = expert_trajs[:num_expert_trajs, :, :] # select first expert_episodes

    if v['obj'] != 'airl':
        expert_samples = expert_trajs.copy().reshape(-1, len(state_indices))
        print(expert_trajs.shape, expert_samples.shape) # ignored starting state
    else: # for airl, we need (s, s') tuples
        expert_samples = expert_trajs
        print(expert_trajs.shape)

    if v['adv_irl']['normalize']:
        expert_samples_ = expert_trajs.copy().reshape(-1, len(state_indices))
        obs_mean, obs_std = expert_samples_.mean(0), expert_samples_.std(0)
        obs_std[obs_std == 0.0] = 1.0 # avoid constant distribution
        expert_samples = (expert_samples - obs_mean) / obs_std # normalize expert data
        print('obs_mean, obs_std', obs_mean, obs_std)
        env_fn = lambda: gym.make(env_name, obs_mean=obs_mean, obs_std=obs_std)
    
    # load expert actions for AIRL
    expert_action_trajs = torch.load(f'expert_data/actions/{env_name}_airl.pt').numpy()
    expert_action_trajs = expert_action_trajs[:num_expert_trajs, :, :] # select first expert_episodes

    # build the discriminator model
    if v['adv_irl']['disc']['model_type'] == 'resnet_disc':
        disc_model = ResNetAIRLDisc(
            len(state_indices),
            device=device,
            **v['adv_irl']['disc']
        ).to(device)
    elif v['adv_irl']['disc']['model_type'] == 'mlp_disc':
        print("using mlp model!")
        disc_model = MLPDisc(
            len(state_indices),
            device=device,
            **v['adv_irl']['disc']
        ).to(device)
        

    sac_agent = SAC(env_fn, None, 
        steps_per_epoch=v['env']['T'],
        max_ep_len=v['env']['T'],
        seed=seed,
        reward_state_indices=state_indices,
        device=device,
        **v['sac']
    )

    algorithm = AdvSMM(
        env_fn=env_fn,
        obj=v['obj'],
        discriminator=disc_model,
        agent=sac_agent,
        state_indices=state_indices,
        target_state_buffer=expert_samples,
        expert_action_trajs=expert_action_trajs,
        device=device,
        logger=logger,
        collect_fn=collect.collect_trajectories_policy_single,
        **v['adv_irl'],
        training_trajs=v['irl']['training_trajs'],
        expert_IS=False,
        v=v
    )

    algorithm.train()    
