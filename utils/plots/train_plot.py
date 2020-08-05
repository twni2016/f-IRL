import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import torch
import numpy as np
from scipy.ndimage import uniform_filter

def print_metrics(metrics):
    info = ""
    for k, v in metrics.items():
        info += f" {k}: {v:.2f}"
    return info

def plot(samples, reward_fn, kde_fn, density_ratio_fn, div: str, output_dir: str, step: int, range_lim: list,
        sac_info: list, measures: list, reward_losses: list, old_reward=None):
    n_pts = 32 # 0.001   

    # construct test points
    test_grid = setup_grid(range_lim, n_pts)

    # plot
    ims = []
    fig, axs = plt.subplots(2, 4, figsize=(30, 12))
    axs = axs.reshape(-1)

    sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps = sac_info
    if sac_test_rets is not None:
        plot_sac_curve(axs[0], sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps)
    ims.append(plot_samples(samples, axs[1], range_lim, n_pts))
    ims.append(plot_density(kde_fn, axs[2], test_grid, n_pts, 'log agent density'))
    plot_cov_curve(axs[3], reward_losses)

    plot_traj(samples, axs[4])
    ims.append(plot_density(density_ratio_fn, axs[5], test_grid, n_pts, f'{div} density ratio', div))
    ims.append(plot_reward_fn(axs[6], test_grid, n_pts, reward_fn))
    ims.append(plot_reward_grad(axs[7], test_grid, n_pts, reward_fn, old_reward))

    # format
    for ax, im in zip([axs[1], axs[2], axs[5], axs[6], axs[7]], ims):
        fig.colorbar(im, ax=ax)
    for idx, ax in enumerate(axs):
        if idx in [0, 3]: continue
        format_ax(ax, range_lim)

    axs[3].set_title(f'{output_dir}\nIRL step {step:d}' + print_metrics(measures))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plt/vis_step_{step:06}.png')) 
    plt.close()

def plot_submission(samples, reward_fn, div: str, output_dir: str, step: int, range_lim: list, 
            measures: list, rho_expert):
    n_pts = 64 # 0.001   

    # construct test points
    test_grid = setup_grid(range_lim, n_pts)

    # plot
    ims = []
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.reshape(-1)
    ims.append(plot_reward_fn(axs[0], test_grid, n_pts, reward_fn))
    ims.append(plot_expert(axs[1], test_grid, n_pts, rho_expert))
    plot_traj(samples, axs[2])
    ims.append(plot_samples(samples, axs[3], range_lim, n_pts))

    # format
    for ax, im in zip([axs[0], axs[1], axs[3]], ims):
        fig.colorbar(im, ax=ax)
    for idx, ax in enumerate(axs):
        format_ax(ax, range_lim)

    axs[3].set_title(f'Method: {div} Step: {step:d} ' + print_metrics(measures))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plt/vis_step_{step:06}.png')) 
    plt.close()

def plot_disc(samples, reward_fn, disc_fn, div:str, output_dir: str, step: int, range_lim: list,
        sac_info: list, disc_loss, measures: list):
    n_pts = 32 # 0.001   

    # construct test points
    test_grid = setup_grid(range_lim, n_pts)

    # plot
    ims = []
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    axs = axs.reshape(-1)

    sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps = sac_info
    if sac_test_rets is not None:
        plot_sac_curve(axs[0], sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps)
    ims.append(plot_samples(samples, axs[1], range_lim, n_pts))
    ims.append(plot_density(disc_fn, axs[2], test_grid, n_pts, 
        'critic value landscape' if div == 'emd' else f'{div} density ratio', div))

    if div == 'emd':
        plot_critic_curve(axs[3], disc_loss)
    else:
        plot_disc_curve(axs[3], disc_loss)

    plot_traj(samples, axs[4])
    ims.append(plot_reward_fn(axs[5], test_grid, n_pts, reward_fn))
    

    # format
    for ax, im in zip([axs[1], axs[2], axs[5]], ims):
        fig.colorbar(im, ax=ax)
    for ax in [axs[1], axs[2], axs[4], axs[5]]:
        format_ax(ax, range_lim)

    axs[-1].set_title(f'{output_dir}\nIRL step {step:d}' + print_metrics(measures))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plt/vis_step_{step:06}.png')) 
    plt.close()

def plot_adv_irl(samples, reward_fn, output_dir: str, step: int, range_lim: list,
        sac_info: list, measures: list):
    n_pts = 32 # 0.001   

    # construct test points
    test_grid = setup_grid(range_lim, n_pts)

    # plot
    ims = []
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.reshape(-1)

    sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps = sac_info
    if sac_test_rets is not None:
        plot_sac_curve(axs[0], sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps)
    ims.append(plot_samples(samples, axs[1], range_lim, n_pts))
    ims.append(plot_reward_fn(axs[2], test_grid, n_pts, reward_fn))
    plot_traj(samples, axs[3])

    # format
    for ax, im in zip(axs[1:3], ims):
        fig.colorbar(im, ax=ax)
    for ax in axs[1:]:
        format_ax(ax, range_lim)

    fkl, rkl, ent = measures
    axs[-1].set_title(f'{output_dir}\nIRL step {step:d}' + print_metrics(measures))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plt/vis_step_{step:06}.png')) 
    plt.close()

def plot_sac_curve(ax, ret, alphas, log_pis, t):
    # print(t)
    # print(ret)
    ax.plot(t, ret)
    ax2 = ax.twinx()
    # ax2.plot(t, alphas, color='red')
    ax2.plot(t, -np.array(log_pis), color='red')
    ax.set_xlabel("Training time steps")
    ax.set_ylabel("Online return")
    ax2.set_ylabel("entropy")
    ax.set_title('alpha = %.2f' % np.mean(alphas))

def plot_disc_curve(ax, loss):
    ax.plot(uniform_filter(loss, 20))
    ax.set_xlabel("Training time steps")
    ax.set_ylabel("BCE logit Loss")
    ax.set_title('Disc loss')

def plot_cov_curve(ax, loss):
    ax.plot(range(1, len(loss)+1), loss, marker='o')
    ax.set_xlabel("Reward grad updates")
    ax.set_ylabel("Reward loss")
    ax.set_title('Reward loss (Cov)')

def plot_critic_curve(ax, loss):
    ax.plot(uniform_filter(loss["total"], 20), label="total loss")
    ax.plot(uniform_filter(loss["main"], 20), label="main loss")
    ax.plot(uniform_filter(loss["grad_pen"], 20), label="grad_pen")

    ax.legend()
    ax.set_xlabel("Training time steps")
    ax.set_ylabel("Loss")
    ax.set_title('EMD Critic loss')

def setup_grid(range_lim, n_pts):
    x = torch.linspace(range_lim[0][0], range_lim[0][1], n_pts)
    y = torch.linspace(range_lim[1][0], range_lim[1][1], n_pts)
    xx, yy = torch.meshgrid((x, y))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz

def format_ax(ax, range_lim):
    ax.set_xlim(range_lim[0][0], range_lim[0][1])
    ax.set_ylim(range_lim[1][0], range_lim[1][1])
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # ax.invert_yaxis()

def plot_samples(samples, ax, range_lim, n_pts):
    s = samples.reshape(-1, samples.shape[2])
    indices = np.random.choice(s.shape[0], size=min(10000, s.shape[0]), replace=False)
    s = s[indices]
    im = ax.hist2d(s[:,0], s[:,1], density=True, norm=LogNorm(),
                    range=range_lim, 
                    bins=n_pts, cmap=plt.cm.jet)
    ax.set_title('SAC Density')
    ax.set_aspect('equal', 'box')
    return im[3] # https://stackoverflow.com/a/42388179/9072850

def plot_traj(samples, ax):
    indices = np.random.choice(samples.shape[0], size=min(100, samples.shape[0]), replace=False)
    s = samples[indices]
    for traj in s:
        ax.plot(traj[:, 0], traj[:, 1])
    ax.set_title('SAC Trajectories')
    ax.set_aspect('equal', 'box')

def plot_expert(ax, test_grid, n_pts, rho_expert, title='Expert Density'):
    xx, yy, zz = test_grid
    rho = rho_expert(zz)

    im = ax.pcolormesh(xx, yy, rho.reshape(n_pts,n_pts), norm=LogNorm(), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return im

def plot_density(density_fn, ax, test_grid, n_pts, title, div=None):
    div in [None, 'fkl', 'rkl', 'js', 'emd']
    xx, yy, zz = test_grid
    log_density = density_fn(zz.numpy())

    if div == 'fkl':
        density = np.exp(log_density)
    elif div == 'js':
        density = softplus(log_density)
    else:
        density = log_density

    # plot
    im = ax.pcolormesh(xx, yy, density.reshape(n_pts, n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return im

def plot_reward_fn(ax, test_grid, n_pts, reward_fn, title='Reward Map'):
    xx, yy, zz = test_grid
    rewards = reward_fn(zz)

    im = ax.pcolormesh(xx, yy, rewards.reshape(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    return im

def plot_reward_grad(ax, test_grid, n_pts, reward_fn, old_reward_fn):
    xx, yy, zz = test_grid
    rewards = reward_fn(zz)
    old_rewards = old_reward_fn(zz)

    diff = rewards - old_rewards

    im = ax.pcolormesh(xx, yy, diff.reshape(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title('Rewards Difference')
    ax.set_aspect('equal', 'box')
    return im

def plot_ratio(samples, ratios, ax):
    idxes = np.random.choice(range(len(samples)), size=min(100, len(samples)))
    sampled_trajs = samples[idxes]
    sampled_ratios = ratios[idxes]

    negative_min = np.min(sampled_ratios)
    positive_max = np.max(sampled_ratios)

    for idx, traj in enumerate(sampled_trajs):
        r = sampled_ratios[idx]
        if r >= 0: # reward should decrease along the trajectory, green
            color = (0, 1, 0, r / positive_max)
        else: # reward should increase along the trajectory, red
            color = (1, 0, 0, r / negative_min)
        ax.plot(traj[:, 0], traj[:, 1], color=color)

    ax.set_ylabel('1 - expert / reward density')
    ax.set_aspect('equal', 'box')

def softplus(x, thres=20):
    return np.where(x > thres, x, np.log(1 + np.exp(x)))
