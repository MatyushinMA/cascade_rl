from a2c_ppo_acktr.model import Policy, AttentionBase, MLPBase
from a2c_ppo_acktr.algo import PPO, A2C_ACKTR
from a2c_ppo_acktr.storage import RolloutStorage

import torch
import gym
import numpy as np

class Unit(object):
    def __init__(self, num_inputs, num_outputs, args=None, is_input=False, is_output=False):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        if is_output:
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(num_outputs,))
        else:
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(num_outputs*args.signal_split,))
        if not (is_input or is_output):
            assert num_inputs == num_outputs
            base = AttentionBase(num_inputs, args.signal_split, args.att_num_heads)
            self.actor_critic = Policy(num_outputs, self.action_space, args.signal_split, base)
        elif is_input:
            base = MLPBase(num_inputs, args.signal_split)
            self.actor_critic = Policy(num_outputs, self.action_space, args.signal_split, base)
        elif is_output:
            base = AttentionBase(num_inputs, 1, args.att_num_heads)
            self.actor_critic = Policy(num_outputs, self.action_space, 1, base)
        self.args = args()
        self.is_input = is_input
        self.is_output = is_output
        self.agent = PPO(
            self.actor_critic,
            self.args.clip_param,
            self.args.ppo_epoch,
            self.args.value_loss_coef,
            self.args.entropy_coef,
            self.args.direction_loss_coef,
            lr=self.args.lr,
            eps=self.args.eps,
            max_grad_norm=self.args.max_grad_norm)
        self.value_losses = []
        self.action_losses = []
        self.dist_entropies = []
        self.direction_losses = []
    
    def __call__(self, state):
        with torch.no_grad():
            value, action, action_log_prob = self.actor_critic.act(torch.from_numpy(state).float())
            return value, action, action_log_prob
            
    def update(self, generator, postwidth=0):
        value_loss, action_loss, dist_entropy, direction_loss = self.agent._update(generator, postwidth=postwidth)
        self.value_losses.append(value_loss)
        self.action_losses.append(action_loss)
        self.dist_entropies.append(dist_entropy)
        self.direction_losses.append(direction_loss)
    
    def clear_stats(self):
        self.value_losses = []
        self.action_losses = []
        self.dist_entropies = []
        self.direction_losses = []