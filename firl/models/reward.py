import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPReward(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_sizes=(256,256),
        hid_act='tanh',
        use_bn=False,
        residual=False,
        clamp_magnitude=10.0,
        device=torch.device('cpu'), 
        **kwargs
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude
        self.input_dim = input_dim
        self.device = device
        self.residual = residual

        self.first_fc = nn.Linear(input_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_bn: block.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        output = self.last_fc(x)
        output = torch.clamp(output, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return output  

    def r(self, batch):
        return self.forward(batch)

    def get_scalar_reward(self, obs):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(obs):
                obs = torch.FloatTensor(obs.reshape(-1,self.input_dim))
            obs = obs.to(self.device)
            reward = self.forward(obs).cpu().detach().numpy().flatten()
        self.train()
        return reward
