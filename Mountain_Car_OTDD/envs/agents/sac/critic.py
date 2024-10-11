#pytorch
import torch
import torch.nn as nn
#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        state = state.to(device) #state to GPU
        action = action.to(device) #action to GPU
        q = self.net( torch.cat([state,action],dim=-1) )
        return q