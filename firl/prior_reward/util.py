import torch
from torch import nn
import torch.nn.functional as F

class Discriminator_reward():
    def __init__(self, discriminator, mode, rew_clip_max=10., state_indices = [0, 1],
        rew_clip_min=-10, reward_scale = 1, device=torch.device("cpu"), agent=None, **kwargs):

        self.discriminator = discriminator
        self.mode = mode
        self.rew_clip_max, self.rew_clip_min, self.reward_scale = rew_clip_max, rew_clip_min, reward_scale
        self.device = device
        self.state_indices = state_indices
        self.agent = agent
        self.state_indices = torch.LongTensor(state_indices).to(device) # for Disciminator observation space

    def disc_forward(self, policy_disc_input, train_disc=True):
        # policy_disc_input: [B, S], where S >= len(self.state_indices)
        # NOTE: sampled from replay buffer is mixture of old policies, but empirically works well. called off-policy training as DAC
        disc_input = torch.index_select(policy_disc_input, 1, self.state_indices)
        disc_logits = self.discriminator(disc_input)

        if train_disc and self.mode == 'airl-disc':
            # do not use sigmoid, instead use special structure depends on policy
            disc_logits = torch.exp(disc_logits)
            # assert agent input is the same dimension as discrim, i.e. no time. cuz expert data no time
            with torch.no_grad():
                _, log_pi = self.agent.ac.pi(policy_disc_input)
            exp_pi = torch.exp(log_pi).unsqueeze(1) # crucial to add dim
            disc_logits = disc_logits / (disc_logits + exp_pi)
            disc_preds = (disc_logits > 0.5).type(disc_logits.data.type())
        else:
            disc_preds = (disc_logits > 0).type(disc_logits.data.type())

        return disc_logits, disc_preds

    def get_reward(self, obs):
        self.discriminator.eval()
        with torch.no_grad():
            disc_logits, _ = self.disc_forward(obs, False) # D' = log(D) - log(1-D)
            disc_logits = disc_logits.view(-1) # must squeeze
        self.discriminator.train()

        # NOTE: important: compute the reward using the algorithm
        if self.mode in ['airl', 'airl-disc']:
            rewards = disc_logits  # NOTE: log(D) - log(1-D) = D' = f(s,a)
        elif self.mode == 'gail':
            rewards = F.softplus(disc_logits, beta=-1) # -log(1-D) = -log(1+e^-D') ignore constant log2
        elif self.mode == 'fairl':
            rewards = torch.exp(disc_logits)*(-1.0*disc_logits) # exp(D')*(-D')

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

    