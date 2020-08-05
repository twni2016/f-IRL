import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
mpl.style.use('seaborn')

load_path = 'potential'

fontsize = 22
def plot(alpha_list, prior_reward_weight_list):
    methods = ['fkl', 'rkl', 'js', 'maxentirl', 'airl', 'gail']
    titles = [r'FKL ($f$-IRL)', r'RKL ($f$-IRL)', r'JS ($f$-IRL)', 'MaxEntIRL', 'f-MAX-RKL', 'GAIL']

    fig= plt.figure(figsize=(2 * 6, 3 * 6 + 5))
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2, 3),
                    axes_pad=0.3,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1
                    )


    for method_idx, method in enumerate(methods):
        alpha_num = len(alpha_list)
        prior_reward_weight_num = len(prior_reward_weight_list)
        ret_array = np.zeros((alpha_num, prior_reward_weight_num + 1))

        for alpha_idx in range(alpha_num):
            alpha = alpha_list[alpha_idx]

            no_prior_ret = np.load(f'./data/prior_reward/{load_path}/no_prior_{alpha}_0.1_sac_test_rets.npy')
            no_prior_ret = np.mean(no_prior_ret[-10:])
            ret_array[alpha_idx][0] = no_prior_ret

            for prior_idx in range(prior_reward_weight_num):
                prior_reward_weight = prior_reward_weight_list[prior_idx]

                ret = np.load(f'./data/prior_reward/{load_path}/{method}_{alpha}_{prior_reward_weight}_sac_test_rets.npy')
                ret = np.mean(ret[-10:])

                ret_array[alpha_idx][prior_idx + 1] = ret

        
        ax = grid[method_idx]

        im = ax.pcolormesh(ret_array, cmap=plt.cm.jet)
        ax.set_facecolor(plt.cm.jet(0.))
        ax.set_title(f'{titles[method_idx]}', fontsize=fontsize)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('prior reward weight', fontsize=fontsize)
        ax.set_ylabel('sac alpha', fontsize=fontsize)
        ax.set_xticks([0.5, len(ret_array) + 0.5])
        ax.set_yticks([0.5, len(ret_array) - 0.5])
        ax.set_xticklabels([0, 0.3])
        ax.set_yticklabels([0.01, 0.3])

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize) 

    # grid[-1].set_axis_off()
    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.ax.tick_params(labelsize=fontsize)

    # plt.savefig(f'./data/prior_reward/{load_path}/image-all.png')
    # plt.tight_layout()
    plt.savefig(f'./tmp-{load_path}.png')
    plt.show()

if __name__ == '__main__':
    alpha_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    prior_reward_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    plot(alpha_list, prior_reward_list)