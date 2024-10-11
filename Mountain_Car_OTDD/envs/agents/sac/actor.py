import numpy as np
#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        state = state.to(device) #state to GPU
        x = self.net(state)
        mu = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2) #limit std
        std = torch.exp(log_std)
        
        pi_distribution = Normal(mu, std)
        action_u = pi_distribution.rsample() #sample actions
        
        #log_likelihood
        log_mu = pi_distribution.log_prob(action_u).sum(axis=-1) 
        log_pi = log_mu - ( 2*( np.log(2) 
                               - action_u 
                               - F.softplus(-2*action_u))).sum(axis=1)
        log_pi = log_pi.unsqueeze(0) #dimension consistency
        
        #actions
        action = torch.tanh(action_u) #bound action [-1,1]
        action = self.act_limit*action #scale actions
        
        return action, log_pi
    
class Actor_Eval(nn.Module): #actor_model
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2,act_limit):
        super(Actor_Eval, self).__init__()
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
        state = state.to(device) #state to GPU
        x = self.net(state)
        mu = self.mean(x)
        return torch.tanh(mu)*self.act_limit