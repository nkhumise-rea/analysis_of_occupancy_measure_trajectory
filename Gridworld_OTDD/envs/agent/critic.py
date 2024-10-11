#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Critic(nn.Module): #critic_model
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states+num_actions,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(),
            nn.Linear(num_hidden_l2,1),
            )
             
    def forward(self, state, action):
        x = torch.cat([state,action],dim=-1)
        return self.net(x)
    
class Critic_Discrete(nn.Module): #critic_model
    def __init__(self,num_states,num_actions,num_hidden_l1):
        super(Critic_Discrete, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_actions),
            )
                     
    def forward(self, state):
        return self.net(state) 