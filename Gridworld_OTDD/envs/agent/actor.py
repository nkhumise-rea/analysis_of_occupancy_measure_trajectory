import numpy as np
#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

class Actor(nn.Module): #actor_model
    def __init__(self,num_states,num_actions,num_hidden_l1,num_hidden_l2,act_limit):
        print('num_actions: ', num_actions)
        xx
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l2),
            nn.ReLU(), 
            )
        
        # self.mean = nn.Linear(num_states,num_actions)
        # self.log_std = nn.Linear(num_states,num_actions)
        self.mean = nn.Linear(num_hidden_l2,num_actions)
        self.log_std = nn.Linear(num_hidden_l2,num_actions)
        self.act_limit = act_limit
          
    def forward(self, state):
        # x = state
        x = self.net(state)
        mu = self.mean(x)
        log_std = self.log_std(x)

        log_std = torch.clamp(log_std,1e-6,1) #avoid -ve exponentials
        std  = torch.exp(log_std)

        pi_distribution = Normal(mu,std)
        action_u = pi_distribution.rsample() #sample actions

        #log_likelihood
        log_mu = pi_distribution.log_prob(action_u)
        log_pi = log_mu - (2*(np.log(2) - action_u - F.softplus(-2*action_u))) 

        #actions
        action = torch.tanh(action_u) #bound action [-1,1]
        action = self.act_limit*action #scale actions

        return action, log_pi
    
class Actor_Discrete(nn.Module): #actor_model
    def __init__(self,num_states,num_actions,num_hidden_l1):
        super(Actor_Discrete, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_actions),
            nn.Softmax(dim=-1), 
            )
        
        #feature_extractor
        self.extractor = nn.Sequential(
            nn.Linear(num_states,num_hidden_l1),
            nn.ReLU(),
            nn.Linear(num_hidden_l1,num_hidden_l1),
            nn.ReLU(),
            )
        
        #classifier/predictor
        self.classifier = nn.Sequential(
            nn.Linear(num_hidden_l1,num_actions),
            nn.Softmax(dim=-1), 
            )

        self.feature = None

    def forward(self, state):
        # action_probs  = self.net(state) #action_probs
        self.feature = self.extractor(state) #features
        action_probs  = self.classifier(self.feature) #action_probs (predictions)

        z = action_probs == 0.0
        z = z.float()*1e-8
        log_action_probs = torch.log(action_probs + z) #log_action_probs

        return action_probs, log_action_probs
    