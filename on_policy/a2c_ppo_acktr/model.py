import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussianRestricted
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, num_outputs, action_space, signal_split, base):
        super(Policy, self).__init__()

        self.base = base
        
        assert num_outputs*signal_split == action_space.shape[0]
        self.num_outputs = num_outputs
        self.signal_split = signal_split
        self.dist = DiagGaussianRestricted(self.base.output_size, action_space.shape[0], low=action_space.low, high=action_space.high)

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        
        action = action.view(self.base.signal_split, self.num_outputs)

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action.view(1, -1))
        dist_entropy = dist.entropy().mean()
        dist_mode = dist.mode()

        return value, action_log_probs, dist_entropy, dist_mode
        

class MLPBase(nn.Module):
    def __init__(self, num_inputs, signal_split, hidden_size=64):
        super(MLPBase, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        self.signal_split = signal_split
        self.hidden_size = hidden_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, signal_split*hidden_size)), nn.Tanh(),
            init_(nn.Linear(signal_split*hidden_size, signal_split*hidden_size)), nn.Tanh())
        
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, signal_split*hidden_size)), nn.Tanh(),
            init_(nn.Linear(signal_split*hidden_size, signal_split*hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(signal_split*hidden_size, 1))

        self.train()
        
    
    @property
    def output_size(self):
        return self.hidden_size*self.signal_split

    def forward(self, inputs):
        x = inputs

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor

class AttentionBase(nn.Module):
    def __init__(self, input_size, signal_split, num_heads, hidden_size=64):
        super(AttentionBase, self).__init__()
        self.hidden_size = hidden_size
        self.signal_split = signal_split
        self.actor_query_w = nn.Parameter(Variable(torch.randn((signal_split, 1, hidden_size), requires_grad=True)))
        self.actor_key_projector = nn.Linear(input_size, hidden_size)
        self.actor_value_projector = nn.Linear(input_size, hidden_size)
        self.actor_mhat = nn.MultiheadAttention(hidden_size, num_heads)
        self.actor_linear = nn.Linear(hidden_size, hidden_size)
        self.actor_act = nn.Tanh()
        
        self.critic_query_w = nn.Parameter(Variable(torch.randn((signal_split, 1, hidden_size), requires_grad=True)))
        self.critic_key_projector = nn.Linear(input_size, hidden_size)
        self.critic_value_projector = nn.Linear(input_size, hidden_size)
        self.critic_mhat = nn.MultiheadAttention(hidden_size, num_heads)
        self.critic_linear = nn.Linear(hidden_size, hidden_size)
        self.critic_act = nn.Tanh()
        self.critic_fin_linear = nn.Linear(hidden_size*signal_split, 1)
        
    @property
    def output_size(self):
        return self.hidden_size*self.signal_split
    
    def forward(self, inputs):
        x = inputs
        query = self.actor_query_w
        key = self.actor_key_projector(x)
        value = self.actor_value_projector(x)
        att = self.actor_mhat(query, key, value)[0]
        proj_att = self.actor_act(self.actor_linear(att))
        actor_hidden = proj_att.flatten()
        
        query = self.critic_query_w
        key = self.critic_key_projector(x)
        value = self.critic_value_projector(x)
        att = self.critic_mhat(query, key, value)[0]
        proj_att = self.critic_act(self.critic_linear(att))
        critic_hidden = proj_att.flatten()
        value = self.critic_fin_linear(proj_att.flatten())
        
        return value, actor_hidden

class DebugAttentionBase(nn.Module):
    def __init__(self, input_size, signal_split, num_heads, hidden_size=64):
        super(DebugAttentionBase, self).__init__()
        self.hidden_size = hidden_size
        self.signal_split = signal_split
        self.actor_query_w = nn.Parameter(Variable(torch.randn((signal_split, 1, hidden_size), requires_grad=True)))
        self.actor_key_projector = nn.Linear(input_size, hidden_size)
        self.actor_value_projector = nn.Linear(input_size, hidden_size)
        self.actor_mhat = nn.MultiheadAttention(hidden_size, num_heads)
        self.actor_linear = nn.Linear(hidden_size, hidden_size)
        self.actor_act = nn.Tanh()
        
        self.critic_query_w = nn.Parameter(Variable(torch.randn((signal_split, 1, hidden_size), requires_grad=True)))
        self.critic_key_projector = nn.Linear(input_size, hidden_size)
        self.critic_value_projector = nn.Linear(input_size, hidden_size)
        self.critic_mhat = nn.MultiheadAttention(hidden_size, num_heads)
        self.critic_linear = nn.Linear(hidden_size, hidden_size)
        self.critic_act = nn.Tanh()
        self.critic_fin_linear = nn.Linear(hidden_size*signal_split, 1)
        
    @property
    def output_size(self):
        return self.hidden_size*self.signal_split
    
    def forward(self, inputs):
        x = inputs
        query = self.actor_query_w
        key = self.actor_key_projector(x)
        value = self.actor_value_projector(x)
        att, weights = self.actor_mhat(query, key, value, need_weights=True)
        print('Actor weights', weights)
        proj_att = self.actor_act(self.actor_linear(att))
        actor_hidden = proj_att.flatten()
        
        query = self.critic_query_w
        key = self.critic_key_projector(x)
        value = self.critic_value_projector(x)
        att, weights = self.critic_mhat(query, key, value, need_weights=True)
        print('Critic weights', weights)
        proj_att = self.critic_act(self.critic_linear(att))
        critic_hidden = proj_att.flatten()
        value = self.critic_fin_linear(proj_att.flatten())
        
        return value, actor_hidden