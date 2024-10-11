import numpy as np
#pytorch
import torch
import torch.nn as nn
#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self,num_states,num_hidden_l1,num_hidden_l2):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,1),
            #nn.Tanh(),           
            )
    
    def forward(self, state):
        state = state.to(device) #state to GPU
        x = self.net(state)
        return x