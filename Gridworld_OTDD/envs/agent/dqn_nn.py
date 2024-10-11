#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hidden):
        super(DQN, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(num_inputs, num_hidden),
        #     nn.ReLU(),
        #     nn.Linear(num_hidden, num_outputs)
        #     )
        
        #feature_extractor
        self.extractor = nn.Sequential(
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            )
        
        #classifier/predictor
        self.classifier = nn.Sequential(
            nn.Linear(num_hidden, num_outputs)
            )
        
    def forward(self, o):
        x = self.extractor(o) #features
        op  = self.classifier(x) #predictions
        return op #x,op
        # return x, self.classifier(x) #features, predictions
        # return self.net(o)