# $f$-IRL: Inverse Reinforcement Learning via State Marginal Matching
Appear in Conference on Robot Learning (CoRL) 2020. This repository is to reproduce the results for our method and baselines showed in the paper.
[Arxiv link], [Website link]

Authors: Tejus Gupta*, Tianwei Ni*, Harshit Sikchi*, Yufei Wang*, Benjamin Eysenbach, Lisa Lee.
where *indicates equal contributions (order by dice rolling)

## Installation
- PyTorch 1.5+
- OpenAI Gym
- [MuJoCo](https://www.roboti.us/license.html)
- `pip install ruamel.yaml` 
- Download expert data that are used in our paper from [Google Drive](https://drive.google.com/drive/folders/1exDW5cyqRIEBmfBW2uRXSFOlJOBdKPtR?usp=sharing) as `expert_data/` folder
  - `states/`: expert state trajectories for each environment. We obtain two sets of state trajectories for our method/MaxEntIRL (`*.pt`) and AIRL (`*_airl.pt`), respectively.
  - `actions/`: expert action trajectories for each environment for AIRL (`*_airl.pt`)
  - `meta/`: expert reward curves through training
  - `reward_models/`: the reward models saved from our algorithm

## File Structure
- f-IRL (Our method): `firl/`
- Baselines (f-MAX, AIRL, BC): `baselines/`
- SAC agent: `common/`
- Environments: `envs/`
- Configurations: `configs/`

## Instructions
- All the experiments are to be run under the root folder. 
- Before starting experiments, please `export PYTHONPATH=${PWD}:$PYTHONPATH` for env variable. 
- We use yaml files in `configs/` for experimental configurations, please change `obj` value (in the first line) for each method, here is the list of `obj` values:
    -  f-IRL (FKL): `fkl`
    -  f-IRL (RKL): `rkl`
    -  f-IRL (JS): `js`
    -  MaxEntIRL: `maxentirl`
    -  f-MAX-RKL: `f-max-rkl`
    -  GAIL: `gail`
    -  AIRL: `airl`
    -  BC: `bc`
- Please keep all the other values in yaml files unchanged to reproduce the results in our paper.
- After running, you will see the training logs in `logs/` folder.

## Experiments
All the commands below are also provided in `run.sh`.

### Sec 5.1 Density task (Reacher)

```bash
# our method and maxentirl. you can vary obj in {`fkl`, `rkl`, `js`, `maxentirl`}
python firl/irl_density.py configs/density/reacher_trace_gauss.yml # Gaussian goal
python firl/irl_density.py configs/density/reacher_trace_mix.yml # Mixture of Gaussians goal
# f-MAX-RKL, GAIL. you can vary obj in {`f-max-rkl`, `gail`}
python baselines/main_density.py configs/density/reacher_trace_gauss.yml # Gaussian goal
python baselines/main_density.py configs/density/reacher_trace_mix.yml # Mixture of Gaussians goal
```

### Sec 5.2 IRL benchmark (MuJoCo)
First, make sure that you have downloaded expert data into `expert_data/`. *Otherwise*, you can generate expert data by training expert policy:
```bash
python common/train_expert.py configs/samples/experts/{env}.yml # env is in {hopper, walker2d, halfcheetah, ant}
```

Then train our method or baseline with provided expert data method (Policy Performance).
Note that you can change the value of `irl: expert_episodes:` into {1, 4, 16} to reproduce the results of {1, 4, 16} trajectories setting shown in Table 3.

```bash
# our method and maxentirl. you can vary obj in {`fkl`, `rkl`, `js`, `maxentirl`}
python firl/irl_samples.py configs/samples/agents/{env}.yml
# baselines
python baselines/bc.py configs/samples/agents/{env}.yml # bc. set obj to `bc`
python baselines/main_samples.py configs/samples/agents/{env}.yml # f-max-rkl. set obj to `f-max-rkl`
python baselines/main_samples.py configs/samples/agents/airl/{env}.yml # airl.
```

After the training is done, you can choose one of the saved reward model to train a policy from scratch (Recovering the Stationary Reward Function).
We provide a learned reward model in `expert_data/reward_models/halfcheetah/` for demonstration purposes.
```bash 
python common/train_optimal.py configs/samples/experts/halfcheetah.yml
```

### Sec 5.3.1 Downstream task 
First, run $f$-IRL or the baselines on the pointmass gridworld with a uniform expert density: 
```bash
# you can change the obj in grid_uniform.yml to be {`fkl`, `rkl`, `js`, `maxentirl`}
python firl/irl_density.py configs/density/grid_uniform.yml 
# you can change the obj in grid_uniform.yml to be {`f-max-rkl`, `gail`}
python baselines/main_density.py configs/density/grid_uniform.yml
```
Then, the discriminator or the reward model should be saved in 
`logs/ContinuousVecGridEnv-v0/{month}-{date}-uniform/{obj}/{detailed-time-stamp}/model/reward_model_*.pkl`

Then update the path to the stored reward model in firl/prior_reward/main.py at line 132-134, and run
```bash
python firl/prior_reward/main.py
```
to test the learned reward on the hard-to-explore task.

The information of the learned sac agent will be saved to 
`data/prior_reward/potential/{save_name}_{alpha}_{prior_reward_weight}_sac_test_rets.npy`

After obtain the learning results from multiple learned rewards/discriminators,  `firl/prior_reward/plot_image.py` and `firl/prior_reward/plot_reward.py` can be used to create figure 4 in the paper.

### Sec 5.3.2 Transfer task
First, make sure that you have downloaded expert data into `expert_data/`. *Otherwise*, you can generate expert data by training expert policy:
Make sure that the `env_name` parameter in `configs/samples/experts/ant_transfer.yml` is set to `CustomAnt-v0`
```bash
python common/train_expert.py configs/samples/experts/ant_transfer.yml
```

Then train our method or baseline with provided expert data method (Policy Performance).
```
python firl/irl_samples.py configs/samples/agents/ant_transfer.yml
```
After the training is done, you can choose one of the saved reward model to train a policy from scratch (Recovering the Stationary Reward Function).

Transferring the reward to disabled Ant:  We provide a learned reward model in `expert_data/reward_models/ant_transfer/` for demonstration purposes.
Make sure that the `env_name` parameter in `configs/samples/experts/ant_transfer.yml` is set to `DisabledAnt-v0`
```bash 
python common/train_optimal.py configs/samples/experts/ant_transfer.yml
```


## Important Note
Parts of the codes are used from the references mentioned below:

- AIRL: https://github.com/justinjfu/inverse_rl in part of `envs/` 
- f-MAX: https://github.com/KamyarGh/rl_swiss/blob/master/run_scripts/adv_smm_exp_script.py in part of `baselines/`
- SAC: https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac in part of `common/sac`
- NPEET: https://github.com/gregversteeg/NPEET in part of `utils/it_estimator.py`

Citation for the papers:
```
@article{ghasemipour2019divergence,
  title={A Divergence Minimization Perspective on Imitation Learning Methods},
  author={Ghasemipour, Seyed Kamyar Seyed and Zemel, Richard and Gu, Shixiang},
  journal={arXiv preprint arXiv:1911.02256},
  year={2019}
}
@article{fu2017learning,
  title={Learning robust rewards with adversarial inverse reinforcement learning},
  author={Fu, Justin and Luo, Katie and Levine, Sergey},
  journal={arXiv preprint arXiv:1710.11248},
  year={2017}
}
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
@article{ver2000non,
  title={Non-parametric entropy estimation toolbox (npeet)},
  author={Ver Steeg, Greg},
  year={2000}
}
```
