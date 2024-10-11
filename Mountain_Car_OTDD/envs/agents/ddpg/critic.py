import numpy as np
#pytorch
import torch
import torch.nn as nn
#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2):
        super(Critic, self).__init__()

        self.state = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            )
        self.action_state = nn.Sequential(
            nn.Linear(num_hidden_l1+num_actions,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,1),
            )
            
    def forward(self, state, action):
        state = state.to(device) #state to GPU
        action = action.to(device) #action to GPU
        x = self.state(state)
        x = self.action_state( torch.cat([x,action],1) )
        return x