import numpy as np
#pytorch
import torch
import torch.nn as nn
#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2):
        super(ActorCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),         
            )
        
        # policy (actor)
        self.mean = nn.Linear(num_hidden_l2,num_actions)
        self.log_std = nn.Linear(num_hidden_l2,num_actions)

        #critic (value)
        self.critic = nn.Linear(num_hidden_l2,1) 

    def forward(self, state):
        state = state.to(device) #state to GPU
        x = self.net(state)
        
        # actor
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2) #limit std
        std = torch.exp(log_std)

        # value
        value = self.critic(x)
        return mean,std,value