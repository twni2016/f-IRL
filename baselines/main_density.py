import sys, os, time
import numpy as np
import math
import gym
from ruamel.yaml import YAML
import envs
from envs.tasks.grid_task import expert_density

import torch
from common.sac import SAC
from baselines.discrim import ResNetAIRLDisc
from baselines.adv_smm import AdvSMM

from utils import system, collect, logger
import datetime
import dateutil.tz
import json

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
    assert v['obj'] in ['f-max-rkl', 'gail', 'fairl', 'airl'] # approximate [RKL, JSD, FKL, RKL]
    assert task_name  in ['uniform', 'gaussian', 'mix_gaussian']
    assert add_time == False
    assert env_name in ["ContinuousVecGridEnv-v0", "ReacherDraw-v0"]

    # logs
    exp_id = f"logs/{env_name}/8-4-{task_name}/{v['obj']}"  # task/obj/date structure
    # exp_id = 'debug'
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)        
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp baselines/main_density.py {log_folder}')
    os.system(f'cp baselines/adv_smm.py {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    print('pid', os.getpid())
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    os.makedirs(os.path.join(log_folder, 'plt'))
    if v['irl']['save_interval'] > 0:
        os.makedirs(os.path.join(log_folder, 'model'))

    env_fn = lambda : gym.make(env_name, **v['env'])
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]

    # rho_expert and samples for FKL estimation (not training)
    rho_expert = expert_density(**v['task'], env=gym_env)
    range_lim = collect.get_range_lim(env_name, v['task'], gym_env)

    if task_name in ['gaussian', 'mix_gaussian']:
        expert_samples = collect.gaussian_samples(env_name, v['task'], gym_env, range_lim)
    elif 'uniform' in task_name:
        expert_samples = collect.expert_samples(env_name, v['task'], rho_expert, range_lim)

    # build the discriminator model
    assert v['adv_irl']['disc']['model_type'] == 'resnet_disc'
    disc_model = ResNetAIRLDisc(
        len(state_indices),
        device=device,
        **v['adv_irl']['disc']
    ).to(device)

    sac_agent = SAC(env_fn, None, 
        add_time=add_time,
        steps_per_epoch=v['env']['T'],
        max_ep_len=v['env']['T'],
        seed=seed,
        reward_state_indices=state_indices, # fix bug
        device=device,
        **v['sac']
    )

    if env_name == "ContinuousVecGridEnv-v0":
        collect_fn = collect.collect_trajectories_policy
    else:
        collect_fn = collect.collect_trajectories_policy_single

    algorithm = AdvSMM(
        env_fn=env_fn,
        obj=v['obj'],
        discriminator=disc_model,
        agent=sac_agent,
        state_indices=state_indices,
        rho_expert=rho_expert,
        target_state_buffer=expert_samples,
        device=device,
        logger=logger,
        collect_fn=collect_fn,
        replay_buffer_size=v['sac']['buffer_size'],
        policy_optim_batch_size=v['sac']['batch_size'],
        **v['adv_irl'],
        **v['density']['kde'],
        training_trajs=v['irl']['training_trajs'],
        v=v,
        range_lim=range_lim,
        expert_IS=True,
    )

    algorithm.train()    