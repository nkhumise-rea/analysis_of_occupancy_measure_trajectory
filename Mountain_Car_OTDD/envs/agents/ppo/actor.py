import numpy as np
#pytorch
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,num_actions),
            nn.Tanh(),           
            )
        self.num_actions = num_actions
        cov_var = torch.full(size=(self.num_actions,), fill_value = 0.5)
        self.cov_mat = torch.diag(cov_var)

    def forward(self, state):
        state = state.to(device) #state to GPU
        mean = self.net(state)

        pi_distribution = MultivariateNormal(mean,self.cov_mat)
        action = pi_distribution.sample() #sample actions
        log_pi = pi_distribution.log_prob(action) #log_prob

        # print('action: ', action)
        # # print('log_pi: ', log_pi.unsqueeze(0))
        # print('log_pi: ', log_pi.unsqueeze(0).detach())
        # xxx
        return action,log_pi.unsqueeze(0).detach() #tensor([[   ]]),tensor([[   ]])
    
    def evaluate(self,state,action):
        state = state.to(device) #state to GPU
        mean = self.net(state)

        pi_distribution = MultivariateNormal(mean,self.cov_mat) #zero_action_sampling
        log_pi = pi_distribution.log_prob(action) #log_prob

        # print('log_pi: ', log_pi)
        # print('log_pi: ', log_pi.unsqueeze(0).detach())
        return log_pi #.unsqueeze(0).detach()
    