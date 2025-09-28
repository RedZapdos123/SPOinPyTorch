#This is the model implementation for the Simple Policy Optimizations algorithm.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#This is the layer initialization function.
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#This is the actor network.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64], is_discrete=True):
        super().__init__()
        self.is_discrete = is_discrete
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        
        if is_discrete:
            self.action_head = layer_init(nn.Linear(input_dim, action_dim), std=0.01)
        else:
            self.mean_head = layer_init(nn.Linear(input_dim, action_dim), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(1, action_dim))
    
    #This is the forward pass.
    def forward(self, x):
        x = self.network(x)
        
        if self.is_discrete:
            return self.action_head(x)
        else:
            mean = self.mean_head(x)
            log_std = self.log_std.expand_as(mean)
            return mean, log_std

#This is the critic network.
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dims=[64, 64]):
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(input_dim, hidden_dim)))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
            
        layers.append(layer_init(nn.Linear(input_dim, 1), std=1.0))
        self.network = nn.Sequential(*layers)
    
    #This is the forward pass.
    def forward(self, x):
        return self.network(x).squeeze(-1)
