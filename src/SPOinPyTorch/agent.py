#This is the implementation of the Simple Policy Optimization (SPO) algorithm as per the research paper: https://arxiv.org/abs/2401.16025

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np


#This is the main class for the SPO agent.
#It takes in the state and action dimensions, the configuration dictionary, and the device to run on.
class SPOAgent:
    def __init__(self, state_dim, action_dim, config, is_discrete=True, action_low=None, action_high=None, device='cpu'):
        self.device = device
        self.is_discrete = is_discrete
        self.action_low = torch.tensor(action_low, device=device) if action_low is not None else None
        self.action_high = torch.tensor(action_high, device=device) if action_high is not None else None
        self.config = config
        
        from .models import Actor, Critic
        
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim, 
            hidden_dims=config["actor_hidden_dims"],
            is_discrete=is_discrete
        ).to(device)
        
        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=config["critic_hidden_dims"]
        ).to(device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config["learning_rate"],
            eps=1e-5
        )

    #This function returns the action, log probability, entropy, and value for the given state. 
    def get_action_and_value(self, state, action=None):
        if self.is_discrete:
            logits = self.actor(state)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(state)
        else:
            mean, log_std = self.actor(state)
            std = torch.exp(log_std)
            probs = Normal(mean, std)
            if action is None:
                action = probs.sample()
            if self.action_low is not None and self.action_high is not None:
                action = torch.clamp(action, self.action_low, self.action_high)
            return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(state)
    
    #This function returns the value for the given state.
    def get_value(self, state):
        return self.critic(state)
    
    #This function computes the Generalized Advantage Estimation (GAE) for the given rewards, dones, values, and next value.
    def compute_gae(self, rewards, dones, values, next_value):
        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_values = next_value
            else:
                next_values = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            
            #Calculate the temporal difference errors.
            delta = rewards[t] + self.config['gamma'] * next_values * next_non_terminal - values[t]
            advantages[t] = lastgaelam = delta + self.config['gamma'] * self.config['gae_lambda'] * next_non_terminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns
    
    #This function updates the actor and critic networks.
    def update(self, states, actions, old_log_probs, advantages, returns):
        self.optimizer.zero_grad()
        
        _, new_log_probs, entropy, values = self.get_action_and_value(states, actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        epsilon = self.config["epsilon"]
        
        #Calculate the SPO objective.
        spo_objective = ratio * advantages - (torch.abs(advantages) / (2 * epsilon)) * torch.pow(ratio - 1.0, 2)
        
        policy_loss = -spo_objective.mean()
        
        value_loss = 0.5 * F.mse_loss(values.squeeze(), returns)
        
        entropy_loss = entropy.mean()
        
        #Calculate the total loss.
        #The formula for the total loss is: L = L_policy + c1 * L_value - c2 * L_entropy.
        total_loss = (
            policy_loss 
            + self.config.get("value_loss_coeff", 0.5) * value_loss 
            - self.config.get("entropy_coeff", 0.0) * entropy_loss
        )
        
        #Backpropagate the loss.
        total_loss.backward()
        if self.config.get("max_grad_norm", None):
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.config["max_grad_norm"]
            )
        
        self.optimizer.step()
        
        #Return the losses.
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
        }
