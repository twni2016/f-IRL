from utils.plots.train_plot import *

# Remove all the visual plots limited to 2D

def plot_disc(div:str, output_dir: str, step: int, 
        sac_info: list, disc_loss, measures: list):
    # plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps = sac_info
    plot_sac_curve(axs[0], sac_test_rets, sac_alphas, sac_log_pis, sac_time_steps)

    if div in ["emd"]:
        plot_critic_curve(axs[1], disc_loss)
    else:
        plot_disc_curve(axs[1], disc_loss)

    axs[-1].set_title(f'{output_dir}\nIRL step {step:d}' + print_metrics(measures))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'plt/vis_step_{step:06}.png')) 
    plt.close()
