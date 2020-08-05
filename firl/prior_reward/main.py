import gym
from envs.vectorized_grid import ContinuousGridEnv
from common.sac import ReplayBuffer, SAC
import torch
from utils import system
import argparse
import numpy as np
from firl.prior_reward.util import Discriminator_reward
import os
from os import path as osp
import json


def use_reward(s):
    return 'rkl' in s or 'fkl' in s or 'js' in s or 'maxentirl' in s

def run(reward_path, alpha=0.01, prior_reward_weight=0.1):
    seed = 0
    system.reproduce(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_kwargs = {
        "T": 30,
        "state_indices": [0, 1],
        'size_x': 6,
        'size_y': 6,
        'prior_reward_weight': prior_reward_weight,
    }

    reward_func = None
    if reward_path is not None:
        dir_path = osp.dirname(osp.dirname(reward_path))
        print("dir path is: ", dir_path)
        v_path = osp.join(dir_path, 'variant.json')
        v = json.load(open(v_path, 'r'))
        print(v)

        if use_reward(v['obj']):
            from firl.models.reward import MLPReward
            reward_kwargs = v['reward']
            reward_func = MLPReward(len(env_kwargs['state_indices']), **reward_kwargs, device=device)
            reward_func.load_state_dict(torch.load(reward_path))
            reward_func.to(device)
        else:
            from firl.models.discrim import ResNetAIRLDisc
            dis_kwargs = v['disc']
            dis_kwargs.update(
                {
                    'state_indices': [0, 1],
                    'rew_clip_min': -10.0, 
                    'rew_clip_max': 10.0,  
                    'reward_scale': 1.0,  
                } 
            )
            discriminator = ResNetAIRLDisc(
                len(env_kwargs['state_indices']),  
                **dis_kwargs,
                device=device
            ).to(device)
            discriminator.load_state_dict(torch.load(reward_path))
            print("discriminator loaded!")
            
            epoch = reward_path[reward_path.rfind('_') + 1: -4]
            print("reward epoch is: ", epoch)
            agent_path = osp.join(dir_path, 'model', f'agent_epoch_{epoch}.pkl')
            fake_env_func = lambda: gym.make(v['env']['env_name'], **v['env'])
            sac_agent = SAC(fake_env_func, None, k=1)
            sac_agent.ac.load_state_dict(torch.load(agent_path))
            print('sac agent loaded!')

            reward_func = Discriminator_reward(discriminator, 
                mode='airl', 
                device=device, 
                agent=sac_agent,
                **dis_kwargs)

    save_name = 'no_prior' if reward_path is None else v['obj']
    if os.path.exists(f'./data/prior_reward/potential/{save_name}_{alpha}_{prior_reward_weight}_sac_test_rets.npy'):
        print("already obtained")
        return

    env = gym.make("GoalGrid-v0")
    reward_func = reward_func.get_scalar_reward if reward_func is not None else None
    env_fn = lambda:gym.make("GoalGrid-v0", r=reward_func, **env_kwargs)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    sac_kwargs = {
    'epochs': 270, # to use AIRL training schedual, change to 60k/30 :2000
    'steps_per_epoch': env_kwargs['T'],
    'log_step_interval': env_kwargs['T'], # to use AIRL training schedual, can change log frequency to be larger, e.g. 300
    'update_every': 1, # update frequency. to use AIRL training schedule, change to 300
    'update_num': 1, # how many update steps at each update time. to use AIRL training schedule, change to 20.
    'random_explore_episodes': 100, # to use AIRL training schedule, change to 35 or 33. roughly 1000 steps.
    'batch_size': 256, # 64
    'lr': 0.003, # 3e-3
    'alpha': alpha,
    'automatic_alpha_tuning': False,
    'reinitialize': True,
    'buffer_size': 12000, 
    }

    replay_buffer = ReplayBuffer(
            state_size, 
            action_size,
            device=device,
            size=sac_kwargs['buffer_size'])

    sac_agent = SAC(env_fn, replay_buffer, 
            update_after=env_kwargs['T'] * sac_kwargs['random_explore_episodes'], 
            max_ep_len=env_kwargs['T'],
            seed=seed,
            start_steps=env_kwargs['T'] * sac_kwargs['random_explore_episodes'],
            reward_state_indices=env_kwargs['state_indices'],
            device=device,
            k=1,
            **sac_kwargs
        )

    if reward_path is not None and 'agent' in reward_path:
        sac_agent.ac.load_state_dict(torch.load(reward_path))
        print("sac agent loaded!")

    sac_test_rets, sac_alphas, sac_log_pis, sac_test_timestep = sac_agent.learn(n_parallel=1, print_out=True)
    if not osp.exists('data/prior_reward/potential'):
        os.makedirs('data/prior_reward/potential')
    np.save(f'./data/prior_reward/potential/{save_name}_{alpha}_{prior_reward_weight}_sac_test_rets.npy', np.asarray(sac_test_rets))
    np.save(f'./data/prior_reward/potential/{save_name}_{alpha}_{prior_reward_weight}_sac_time_steps.npy', np.asarray(sac_test_timestep))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_path', type=str, default=None)
    args = parser.parse_args()

    reward_paths = [
        None,
        # 'logs/ContinuousVecGridEnv-v0/8-4-uniform/fkl/2020_08_04_23_14_32/model/reward_model_0.pkl',
    ]

    alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    prior_reward_weights = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    for reward_path in reward_paths:
        for alpha in alphas:
            if reward_path is None:
                if os.path.exists(f'./data/prior_reward/5-31/no_prior_{alpha}_0.1_sac_test_rets.npy'):
                    print("already obtained")
                    continue
                run(reward_path, alpha)
            else:
                for prior_reward_weight in prior_reward_weights:
                    print(f"reward_path {reward_path} alpha {alpha} prior_reward_weight {prior_reward_weight}")
                    run(reward_path, alpha, prior_reward_weight)

