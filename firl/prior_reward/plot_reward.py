import gym
from common.sac import ReplayBuffer, SAC
import torch
from utils import system
import argparse
import numpy as np
from firl.prior_reward.util import Discriminator_reward
import os
from os import path as osp
import json
from matplotlib import pyplot as plt
import matplotlib

matplotlib.style.use('seaborn')

def use_reward(s):
    return 'rkl' in s or 'fkl' in s or 'js' in s or 'maxentirl' in s

def setup_grid(range_lim, n_pts):
    x = torch.linspace(range_lim[0], range_lim[1], n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz

def plot_reward(reward_paths):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_kwargs = {
        "T": 30,
        "state_indices": [0, 1],
        'size_x': 6,
        'size_y': 6,
    }

    fig, axs = plt.subplots(2, 3, figsize=(2 * 6, 3 * 6 + 5))
    axs = axs.reshape(-1)

    fontsize = 20
    titles = ['FKL ($f$-IRL)', 'RKL ($f$-IRL)', 'JS ($f$-IRL)', 'MaxEntIRL', 'f-MAX_RKL', 'GAIL']
    for reward_idx, reward_path in enumerate(reward_paths):
        reward_func = None
        if reward_path is not None:
            dir_path = osp.dirname(osp.dirname(reward_path))
            print("dir path is: ", dir_path)
            v_path = osp.join(dir_path, 'variant.json')
            v = json.load(open(v_path, 'r'))
            print(v)

            if use_reward(v['obj']):
                from continuous.prior_reward.old_models import MLPReward
                reward_kwargs = v['reward']
                reward_func = MLPReward(len(env_kwargs['state_indices']), **reward_kwargs, device=device)
                reward_func.load_state_dict(torch.load(reward_path))
            else:
                from continuous.prior_reward.old_models import ResNetAIRLDisc
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

        n_pts = 100 # 0.001
        range_lim = [0, 6]

        # construct test points
        ax = axs[reward_idx]
        test_grid = plot.setup_grid(range_lim, n_pts)
        im = plot.plot_reward_fn(ax, test_grid, n_pts, 100, reward_func.get_scalar_reward)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.03)
        cbar = fig.colorbar(im, cax=cax)

        cbar.ax.tick_params(labelsize=fontsize) 
        cbar.ax.tick_params(pad=0.05)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 

        ax.set_title(titles[reward_idx], fontsize=fontsize)
        if reward_idx % 3 != 0:
            # ax.get_yaxis().set_ticks([])
            ax.get_yaxis().set_visible(False)

    # axs[-1].set_axis_off()
    plt.tight_layout()
    plt.savefig('./reward.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_path', type=str, default=None)
    args = parser.parse_args()

    reward_paths = [
        None
    ]

    plot_reward(reward_paths)
