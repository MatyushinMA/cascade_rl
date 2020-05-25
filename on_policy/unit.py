from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.algo import PPO, A2C_ACKTR
from a2c_ppo_acktr.storage import RolloutStorage

import torch
import gym
import numpy as np

class Unit(object):
    def __init__(self, num_inputs, num_outputs, postcode=None, args=None, critic=None, action_space=None):
        self.num_inputs = num_inputs
        if action_space is None:
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(num_outputs,))
        else:
            self.action_space = action_space
        self.postcode = postcode
        self.actor_critic = Policy((self.num_inputs,), self.action_space, base_kwargs={'recurrent' : False,
                                                                                       'critic' : critic,
                                                                                       'postcode' : self.postcode})
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
        self.memory = RolloutStorage(self.args.memory_capacity, 1,
                                     (self.num_inputs,), self.action_space,
                                     self.actor_critic.recurrent_hidden_state_size)
        self.value_losses = []
        self.action_losses = []
        self.dist_entropies = []
        self.rewards = []
        self._rewards = []
    
    def __call__(self, state, put=True):
        with torch.no_grad():
            if put:
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
            else:
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                            torch.from_numpy(state).float().view(1, -1),
                            None,
                            None)
            return value, action, action_log_prob, recurrent_hidden_states
    
    def reward(self, reward):
        self.memory.rewards[self.memory.step - 1] += reward
        self._rewards.append(reward)
    
    def reset(self):
        self.rewards.append(sum(self._rewards))
        self._rewards = []
    
    def done(self):
        self.memory.masks[self.memory.step] = 0
        
    def _update(self, generator):
        value_loss, action_loss, dist_entropy = self.agent._update(generator)
        self.value_losses.append(value_loss)
        self.action_losses.append(action_loss)
        self.dist_entropies.append(dist_entropy)
    
    def update(self):
        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                                self.memory.obs[self.memory.step],
                                self.memory.recurrent_hidden_states[self.memory.step],
                                self.memory.masks[self.memory.step]).detach()
        self.memory.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                               self.args.gae_lambda, self.args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = self.agent.update(self.memory)
        self.value_losses.append(value_loss)
        self.action_losses.append(action_loss)
        self.dist_entropies.append(dist_entropy)
        self.memory.after_update()
    
    def clear_memory(self):
        self.memory = RolloutStorage(self.args.memory_capacity, 1,
                                     (self.num_inputs,), self.action_space,
                                     self.actor_critic.recurrent_hidden_state_size)
    
    def clear_stats(self):
        self.value_losses = []
        self.action_losses = []
        self.dist_entropies = []
        self.rewards = []