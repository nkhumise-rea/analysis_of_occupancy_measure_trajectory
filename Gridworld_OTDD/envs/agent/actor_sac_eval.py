import numpy as np
#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Actor(nn.Module): #actor_model
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2,act_limit):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(), 
            )

        self.mean = nn.Linear(num_hidden_l2,num_actions)
        self.log_std = nn.Linear(num_hidden_l2,num_actions)
        self.act_limit = act_limit
          
    def forward(self, state):
        x = self.net(state)
        mu = self.mean(x)

        return torch.tanh(mu)*self.act_limit