# '''
# Code built on top of https://github.com/KamyarGh/rl_swiss 
# Refer[Original Code]: https://github.com/KamyarGh/rl_swiss
# '''
# rl_swiss/rlkit/torch/state_marginal_matching/adv_smm.py
# rl_swiss/rlkit/core/base_algorithm.py

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F
from common.sac import ReplayBuffer
from utils import collect, eval
from utils.plots import train_plot
from sklearn import neighbors
from matplotlib import pyplot as plt
import os, copy
import os.path as osp

class AdvSMM:
    def __init__(
        self,
        env_fn, 
        obj, # f-max-rkl, gail, fairl, airl
        discriminator,
        agent, # e.g. SAC
        state_indices, # omit timestamp
        target_state_buffer, # from sampling method (e.g. rejection sampling)
        logger,
        collect_fn,
        replay_buffer_size,
        policy_optim_batch_size,
        
        num_epochs=400,
        num_steps_per_epoch=60000,
        num_steps_between_train_calls=1, # 2400,
        min_steps_before_training=1200, 
        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=20,
        num_policy_updates_per_loop_iter=5,
        num_initial_disc_iters=100,

        expert_IS=True,
        airl_rew_shaping=True,
        rho_expert=None,
        disc_optim_batch_size=800,
        disc_lr=0.0003,
        disc_momentum=0.0,
        disc_optimizer_class=optim.Adam,
        use_grad_pen=True,
        grad_pen_weight=1.0, # NOTE

        reward_scale=0.25,  # NOTE this is 1/alpha
        rew_clip_min=None,
        rew_clip_max=None,

        training_trajs = 200,
        bandwidth=0.2, # may be adaptive by cross validation
        kernel='epanechnikov',

        wrap_absorbing=False, # DAC
        device=torch.device("cpu"),

        save_interval = 50,
        eval_interval = 20, # in terms of discrmintor update
        v=None,
        range_lim=None,

        gamma = 0.99, # gamma for airl disc logits computation
        expert_action_trajs=None, # AIRL requires expert actions

        **kwargs   # for redundant arguments 
    ):  
        self.mode = obj
        assert self.mode in ['f-max-rkl', 'gail', 'fairl', 'airl'], 'Invalid adversarial irl algorithm!'
        if self.mode == 'airl':
            assert expert_IS is False, "airl needs expert samples!"
            # assert use_grad_pen is False, 'airl does not use gradient penalty!'

        self.env, self.test_env = env_fn(), env_fn()
        self.device = device
        self.logger = logger
        self.collect_fn = collect_fn

        self.state_indices = torch.LongTensor(state_indices).to(device) # for Disciminator observation space
        self.rho_expert = rho_expert
        self.target_state_buffer = self.process_target_state_buffer(target_state_buffer, expert_action_trajs)
        self.agent = agent
        self.save_interval = save_interval
        self.range_lim = range_lim

        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(
                                self.env.observation_space.shape[0], 
                                self.env.action_space.shape[0],
                                device=device,
                                size=replay_buffer_size)

        # discriminator
        if self.mode != 'airl': # f-max
            self.discriminator = discriminator
            self.disc_optimizer = disc_optimizer_class(
                self.discriminator.parameters(),
                lr=disc_lr,
                betas=(disc_momentum, 0.999)
            )
            self.bce = nn.BCEWithLogitsLoss().to(device)
        else: # AIRL
            self.reward_model = discriminator
            self.value_model = copy.deepcopy(discriminator)
            self.disc_optimizer = disc_optimizer_class(
                list(self.reward_model.parameters()) + list(self.value_model.parameters()),
                lr=disc_lr,
                betas=(disc_momentum, 0.999)
            )
            self.gamma = gamma
            self.bce = nn.BCELoss().to(device)        

        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1)
            ],
            dim=0
        ).to(device)

        self.disc_optim_batch_size = disc_optim_batch_size
        self.policy_optim_batch_size = policy_optim_batch_size

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight
        self.airl_rew_shaping = airl_rew_shaping

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.reward_scale = reward_scale
        if self.mode == 'airl':
            self.reward_scale = 1.0
            self.rew_clip_min = self.rew_clip_max = None

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self.expert_IS = expert_IS
        self.not_done_initial_disc_iters = not expert_IS
        # remove initial training of discriminator, we do not have expert samples anyway.
        if not self.expert_IS:
            self.max_real_return_det, self.max_real_return_sto = -np.inf, -np.inf

        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_steps_between_train_calls = num_steps_between_train_calls
        self.max_path_length = self.agent.max_ep_len
        self.min_steps_before_training = min_steps_before_training
        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter
        self.num_initial_disc_iters = num_initial_disc_iters

        # evaluation
        self.training_trajs = training_trajs
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.eval_interval = eval_interval
        self.v = v

    def train(self):
        self.try_evaluate("Running", 0)
        # based on common/sac.py learn()
        o, ep_len = self.env.reset(), 0

        for epoch in range(1, self.num_epochs+1):
            for steps_this_epoch in range(self.num_steps_per_epoch):
                a = self.agent.get_action(o) # now assumes no early random exploration

                o2, _, d, _ = self.env.step(a) # reward is set by discrimator, not here

                ep_len += 1
                self._n_env_steps_total += 1
                # d = False # keep false for pointmass env. if ep_len==self.max_path_length else d
                d = False if ep_len==self.max_path_length else d

                self.replay_buffer.store(o, a, 0.0, o2, d)
                o = o2

                if d or (ep_len==self.max_path_length):
                    o, ep_len = self.env.reset(), 0

                if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                    self._try_to_train(epoch)

            self.try_evaluate("Running", epoch)

            if self.expert_IS and self.save_interval > 0 and epoch % self.save_interval == 0:
                save_name = osp.join(self.logger.get_dir(), f"model/disc_epoch_{epoch}.pkl")
                torch.save(self.discriminator.state_dict(), save_name)
                # save_name = osp.join(self.logger.get_dir(), f"model/agent_epoch_{epoch}.pkl")
                # torch.save(self.agent.ac.state_dict(), save_name)

    def _can_train(self):
        return self.replay_buffer.size >= self.min_steps_before_training

    def _try_to_train(self, epoch):
        if self._can_train():
            self.training_mode(True)
            self._do_training(epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)

    def _do_training(self, epoch):
        if self.not_done_initial_disc_iters:
            # disc_stat = np.zeros((3,))
            for _ in range(self.num_initial_disc_iters):
                self._do_reward_training(epoch) # disc_stat += 

            self.not_done_initial_disc_iters = False

        disc_stat = np.zeros((3,))
        policy_stat = np.zeros((7,))
        for t in range(self.num_update_loops_per_train_call): # assume is 1
            for _ in range(self.num_disc_updates_per_loop_iter):
                disc_stat += self._do_reward_training(epoch)
            for _ in range(self.num_policy_updates_per_loop_iter):
                policy_stat += self._do_policy_training(epoch)

        if self.expert_IS:
            disc_stat /= self.num_update_loops_per_train_call * self.num_disc_updates_per_loop_iter
            policy_stat /= self.num_update_loops_per_train_call * self.num_policy_updates_per_loop_iter
            self.record_discrim(disc_stat)
            self.record_policy(policy_stat)
            self.logger.record_tabular("epoch", epoch)
            self.logger.record_tabular("train_step", self._n_train_steps_total)
            self.logger.dump_tabular()

    def process_target_state_buffer(self, target_state_buffer, expert_action_trajs):
        if self.mode != 'airl':
            return target_state_buffer
        else:
            assert len(target_state_buffer) == len(expert_action_trajs)
            obs = []
            actions = []
            obs2 = []
            for idx in range(len(target_state_buffer)):
                obs.append(target_state_buffer[idx][:-1]) # s1 -- sT-1
                actions.append(expert_action_trajs[idx][1:]) # a1 -- aT-1
                obs2.append(target_state_buffer[idx][1:]) # s2 -- sT


            obs = np.concatenate(obs, axis=0)
            actions = np.concatenate(actions, axis=0)
            obs2 = np.concatenate(obs2, axis=0)
            return (obs, actions, obs2)

    def get_target_batch(self, batch_size):
        if self.mode != 'airl':
            batch = self.target_state_buffer[np.random.choice(self.target_state_buffer.shape[0], size=batch_size)]
            return torch.FloatTensor(batch).to(self.device)
        else:
            obs, actions, obs2 = self.target_state_buffer
            indices = np.random.choice(len(obs), size=batch_size)
            return torch.FloatTensor(obs[indices]).to(self.device), \
                    torch.FloatTensor(actions[indices]).to(self.device), \
                    torch.FloatTensor(obs2[indices]).to(self.device)


    def _do_reward_training(self, epoch):
        '''
            Train the discriminator
        '''
        self.disc_optimizer.zero_grad()

        if self.expert_IS: # assume we can only use policy states
            # NOTE: only apply to f-max

            policy_disc_input = self.replay_buffer.sample_batch(self.disc_optim_batch_size)['obs']

            # fit an agent density, using all samples in replay buffer
            agent_density = neighbors.KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            agent_samples = self.replay_buffer.state[:, self.state_indices.cpu().numpy().astype(np.uint8)]
            agent_density.fit(agent_samples)

            # get the importance sampling weight for the expert stat
            policy_input = torch.index_select(policy_disc_input, 1, self.state_indices).cpu().numpy()
            agent_sample_density = np.exp(agent_density.score_samples(policy_input))
            expert_density = self.rho_expert(policy_input)
            is_ratio = expert_density / agent_sample_density # importance sampling for the expert loss
            agent_ratio = np.ones((self.disc_optim_batch_size)) # just all 1.
            weights = np.concatenate([is_ratio, agent_ratio])
            weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)

            # create a new bce loss, where the expert part has the importance sampling weight
            bce = nn.BCEWithLogitsLoss(weight=weights).to(self.device)
            disc_logits, _ = self.disc_forward(policy_disc_input) 
            
            disc_logits = torch.cat([disc_logits, disc_logits], dim=0) 

            disc_ce_loss = bce(disc_logits, self.bce_targets)
            # accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()
            disc_grad_pen_loss = 0.0
        
        else:
            if self.mode != 'airl': # standard f-max
                policy_disc_input = self.replay_buffer.sample_batch(self.disc_optim_batch_size)['obs']
                expert_disc_input = self.get_target_batch(self.disc_optim_batch_size) # access to expert samples
                disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0) # (2*B, S)

                disc_logits, _ = self.disc_forward(disc_input)
                disc_ce_loss = self.bce(disc_logits, self.bce_targets)

                if self.use_grad_pen: # gradient penalty
                    eps = torch.rand((self.disc_optim_batch_size, 1)).to(self.device)
                    
                    interp_obs = eps*expert_disc_input + (1-eps)*policy_disc_input # interpolate
                    interp_obs = interp_obs.detach()
                    interp_obs.requires_grad_(True)

                    gradients = autograd.grad(
                        outputs=self.disc_forward(interp_obs)[0].sum(),
                        inputs=[interp_obs],
                        create_graph=True, retain_graph=True, only_inputs=True
                    )[0]       # gradients w.r.t. inputs (instead of parameters)
                    
                    # GP from Gulrajani et al. https://arxiv.org/pdf/1704.00028.pdf (WGAN-GP)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                    disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

                    # # GP from Mescheder et al. https://arxiv.org/pdf/1801.04406.pdf
                    # gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean()
                    # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
                else:
                    disc_grad_pen_loss = 0.0

            else: # airl
                sampled_batch = self.replay_buffer.sample_batch(self.disc_optim_batch_size)
                policy_state = sampled_batch['obs']
                policy_action = sampled_batch['act']
                policy_next_state = sampled_batch['obs2']

                expert_state, expert_action, expert_next_state = \
                    self.get_target_batch(self.disc_optim_batch_size)
            
                disc_logits, _ = self.disc_forward_airl(
                        policy_state, policy_action, policy_next_state,
                        expert_state, expert_action, expert_next_state
                    )
                disc_ce_loss = self.bce(disc_logits, self.bce_targets)

                if self.use_grad_pen: # gradient penalty on reward and value separately
                    # grad pen on reward
                    eps = torch.rand((self.disc_optim_batch_size, 1)).to(self.device)
                    interp_obs = eps*expert_state + (1-eps)*policy_state # interpolate
                    interp_obs = interp_obs.detach()
                    interp_obs.requires_grad_(True)
                    gradients = autograd.grad(
                        outputs=self.reward_model(interp_obs).sum(),
                        inputs=[interp_obs],
                        create_graph=True, retain_graph=True, only_inputs=True
                    )[0]       # gradients w.r.t. inputs (instead of parameters)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    # grad pen on value
                    eps2 = torch.rand((self.disc_optim_batch_size, 1)).to(self.device)
                    interp_obs2 = eps2*expert_state + (1-eps2)*policy_state # use either cur or next state
                    interp_obs2 = interp_obs2.detach()
                    interp_obs2.requires_grad_(True)
                    gradients2 = autograd.grad(
                        outputs=self.value_model(interp_obs2).sum(),
                        inputs=[interp_obs2],
                        create_graph=True, retain_graph=True, only_inputs=True
                    )[0]       # gradients w.r.t. inputs (instead of parameters)
                    gradient_penalty += ((gradients2.norm(2, dim=1) - 1) ** 2).mean()

                    disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight
                else:
                    disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer.step()

        # return disc stat
        return np.array([disc_total_loss.item(), disc_ce_loss.item(), disc_total_loss.item() - disc_ce_loss.item()])


    def _do_policy_training(self, epoch):
        policy_batch = self.replay_buffer.sample_batch(self.policy_optim_batch_size)
        obs, obs2 = policy_batch['obs'], policy_batch['obs2']
        policy_batch['rew'] = self.get_reward(obs, obs2)

        # policy optimization step
        agent_stat = self.agent.update(policy_batch) # loss-q, loss-pi, log-pi. 
        # in original f-MAX, it adds regularization on policy mean & logstd

        reward_stat = np.array([policy_batch['rew'].mean().item(), policy_batch['rew'].std().item(), 
                                policy_batch['rew'].max().item(), policy_batch['rew'].min().item()])
        return np.concatenate((agent_stat, reward_stat)) # 7-dim

    def disc_forward_airl(self, p_obs, p_act, p_obs_2, e_obs, e_act, e_obs_2):
        obs = torch.cat([e_obs, p_obs], dim=0)
        act = torch.cat([e_act, p_act], dim=0)
        obs_2 = torch.cat([e_obs_2, p_obs_2], dim=0)

        reward = self.reward_model(obs)
        cur_val = self.value_model(obs)
        next_val = self.value_model(obs_2)

        log_p = reward + self.gamma * next_val - cur_val
        with torch.no_grad():
            log_q = self.agent.ac.log_prob(obs, act) # (B,)
            log_q = log_q.unsqueeze(1)
            baseline = torch.max(log_p, log_q)

        log_p -= baseline
        log_q -= baseline
        disc_logits = torch.exp(log_p) / (torch.exp(log_p) + torch.exp(log_q))
        disc_preds = (disc_logits > 0.5).type(disc_logits.data.type())
        
        return disc_logits, disc_preds 

    def disc_forward(self, policy_disc_input):
        # policy_disc_input: [B, S], where S >= len(self.state_indices)
        # NOTE: sampled from replay buffer is mixture of old policies, but empirically works well. called off-policy training as DAC
        if policy_disc_input.shape[1] > len(self.state_indices):
            disc_input = torch.index_select(policy_disc_input, 1, self.state_indices)
        else:
            disc_input = policy_disc_input # for get_scalar_reward() plot

        disc_logits = self.discriminator(disc_input) # (B, 1)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())

        return disc_logits, disc_preds

    def get_reward(self, obs, obs2=None):
        if self.mode != 'airl':
            self.discriminator.eval()
            with torch.no_grad():
                disc_logits, _ = self.disc_forward(obs) # D' = log(D) - log(1-D)
                disc_logits = disc_logits.view(-1) # must squeeze
            self.discriminator.train()

            # NOTE: important: compute the reward using the algorithm
            if self.mode in ['f-max-rkl']:
                rewards = disc_logits  # NOTE: log(D) - log(1-D) = D' = f(s,a)
            elif self.mode == 'gail':
                rewards = F.softplus(disc_logits, beta=-1) # -log(1-D) = -log(1+e^-D') ignore constant log2
            elif self.mode == 'fairl':
                rewards = torch.exp(disc_logits)*(-1.0*disc_logits) # exp(D')*(-D')
        else:
            self.reward_model.eval()
            with torch.no_grad():
                rewards = self.reward_model(obs)
                if self.airl_rew_shaping:
                    rewards += self.gamma * self.value_model(obs2) - self.value_model(obs)
                rewards = rewards.view(-1) # must squeeze
            self.reward_model.train()

        if self.rew_clip_max is not None:
            rewards = torch.clamp(rewards, max=self.rew_clip_max)
        if self.rew_clip_min is not None:
            rewards = torch.clamp(rewards, min=self.rew_clip_min)

        rewards *= self.reward_scale
        return rewards

    def get_scalar_reward(self, obs):
        # obs is mesh numpy array
        if not torch.is_tensor(obs):
            obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)
        rewards = self.get_reward(obs)
        return rewards.cpu().numpy().flatten()

    @property
    def networks(self):
        if self.mode != 'airl':
            return [self.discriminator] + self.agent.networks
        else:
            return [self.reward_model, self.value_model] + self.agent.networks

    def training_mode(self, mode): # mainly for batch1D in Discrim
        for net in self.networks:
            net.train(mode)

    def record_discrim(self, disc_stat):
        self.logger.record_tabular("disc_total_loss", round(disc_stat[0], 4))
        self.logger.record_tabular("disc_bce",        round(disc_stat[1], 4))
        self.logger.record_tabular("disc_gp",         round(disc_stat[0] - disc_stat[1], 4))
        self.logger.record_tabular("disc_acc",        round(disc_stat[2], 4))

    def record_policy(self, policy_stat):
        self.logger.record_tabular("loss_q",   round(policy_stat[0], 4))
        self.logger.record_tabular("loss_pi",  round(policy_stat[1], 4))
        self.logger.record_tabular("log_pi",   round(policy_stat[2], 4))
        self.logger.record_tabular("rew_mean", round(policy_stat[3], 4))
        self.logger.record_tabular("rew_std",  round(policy_stat[4], 4))
        self.logger.record_tabular("rew_max",  round(policy_stat[5], 4))
        self.logger.record_tabular("rew_min",  round(policy_stat[6], 4))

    def try_evaluate(self, policy_type: str, epoch):
        assert policy_type == "Running"
        update_time = self._n_train_steps_total * self.num_update_loops_per_train_call * self.num_disc_updates_per_loop_iter
        env_steps = self._n_env_steps_total

        sac_info = [None, None, None, None]
        samples = self.collect_fn(self.test_env, self.agent, n = self.training_trajs, 
            state_indices=self.state_indices.detach().cpu().numpy())
        agent_emp_states = samples[0]

        expert_samples = self.target_state_buffer[0] if self.mode == 'airl' else self.target_state_buffer
        metrics = eval.KL_summary(expert_samples, agent_emp_states.reshape(-1, agent_emp_states.shape[2]), 
                                env_steps, policy_type, self.v['task']['task_name'] == 'uniform' if self.expert_IS else False)

        if not self.expert_IS:
            real_return_det = eval.evaluate_real_return(self.agent.get_action, self.test_env, 
                                        self.v['irl']['eval_episodes'], self.v['env']['T'], True)
            metrics["Real Det Return"] = real_return_det
            print(f"real det return avg: {real_return_det:.2f}")
            self.logger.record_tabular("Real Det Return", round(real_return_det, 2))

            real_return_sto = eval.evaluate_real_return(self.agent.get_action, self.test_env, 
                                        self.v['irl']['eval_episodes'], self.v['env']['T'], False)
            metrics["Real Sto Return"] = real_return_sto
            print(f"real sto return avg: {real_return_sto:.2f}")
            self.logger.record_tabular("Real Sto Return", round(real_return_sto, 2))

            if real_return_det > self.max_real_return_det and real_return_sto > self.max_real_return_sto:
                self.max_real_return_det, self.max_real_return_sto = real_return_det, real_return_sto
                save_name = osp.join(self.logger.get_dir(), 
                    f"model/disc_epoch{epoch}_det{self.max_real_return_det:.0f}_sto{self.max_real_return_sto:.0f}.pkl")
                if self.mode != 'airl':
                    torch.save(self.discriminator.state_dict(), save_name)
                else:
                    torch.save(self.reward_model.state_dict(), save_name)

        self.logger.record_tabular(f"{policy_type} Update Time", update_time)
        self.logger.record_tabular(f"{policy_type} Env Steps", env_steps)
        self.logger.dump_tabular()

        if self.expert_IS:
            if self.v['env']['env_name'] == 'ReacherDraw-v0':
                train_plot.plot_submission(agent_emp_states, self.get_scalar_reward, self.mode,
                self.logger.get_dir(), env_steps, self.range_lim, metrics, self.rho_expert)
            else:
                train_plot.plot_adv_irl(agent_emp_states, self.get_scalar_reward, 
                                    self.logger.get_dir(), env_steps, self.range_lim,
                                    sac_info, metrics)
