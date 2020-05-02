from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.algo import PPO
from a2c_ppo_acktr.storage import RolloutStorage

import torch
import gym
import numpy as np

class Unit(object):
    def __init__(self, num_inputs, num_outputs, args=None):
        self.num_inputs = num_inputs
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(num_outputs,))
        self.actor_critic = Policy((self.num_inputs,), self.action_space, base_kwargs={'recurrent' : False})
        if not args:
            class args(object):
                eval_interval = None
                log_interval = 10
                use_gae = False
                num_env_steps = 10e6
                num_steps = 32
                clip_param = 0.2
                ppo_epoch = 4
                num_mini_batch = 32
                value_loss_coef = 0.5
                entropy_coef = 0.01
                lr = 7e-4
                eps = 1e-5
                max_grad_norm = 0.5
                gamma = 0.99
                gae_lambda = 0.95
                use_proper_time_limits = False
        self.args = args()
        self.agent = PPO(
            self.actor_critic,
            self.args.clip_param,
            self.args.ppo_epoch,
            self.args.num_mini_batch,
            self.args.value_loss_coef,
            self.args.entropy_coef,
            lr=self.args.lr,
            eps=self.args.eps,
            max_grad_norm=self.args.max_grad_norm)
        self.memory = RolloutStorage(self.args.num_steps, 1,
                                     (self.num_inputs,), self.action_space,
                                     self.actor_critic.recurrent_hidden_state_size)
        self.value_losses = []
        self.action_losses = []
        self.dist_entropies = []
    
    def __call__(self, state):
        with torch.no_grad():
            self.memory.obs[self.memory.step].copy_(torch.from_numpy(state).float())
            value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        self.memory.obs[self.memory.step],
                        self.memory.recurrent_hidden_states[self.memory.step],
                        self.memory.masks[self.memory.step])
            self.memory.insert(torch.zeros(self.num_inputs),
                               recurrent_hidden_states,
                               action,
                               action_log_prob,
                               value,
                               torch.FloatTensor([[0]]),
                               torch.FloatTensor([[1]]),
                               torch.FloatTensor([[1]]))
        return action[0].numpy()
    
    def reward(self, reward):
        self.memory.rewards[self.memory.step - 1] += reward
    
    def done(self):
        self.memory.masks[self.memory.step] = 0
    
    def update(self):
        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                                self.memory.obs[-1],
                                self.memory.recurrent_hidden_states[-1],
                                self.memory.masks[-1]).detach()
        self.memory.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                               self.args.gae_lambda, self.args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = self.agent.update(self.memory)
        self.value_losses.append(value_loss)
        self.action_losses.append(action_loss)
        self.dist_entropies.append(dist_entropy)
        self.memory.after_update()
    
    def clear_memory(self):
        self.memory = RolloutStorage(self.args.num_steps, 1,
                                     (self.num_inputs,), self.action_space,
                                     self.actor_critic.recurrent_hidden_state_size)
    
    def clear_stats(self):
        self.value_losses = []
        self.action_losses = []
        self.dist_entropies = []