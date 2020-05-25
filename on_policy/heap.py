import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax, expit as sigmoid
import numpy as np
import numpy.random as npr
from annoy import AnnoyIndex
import os
import torch

from unit import Unit
from a2c_ppo_acktr.model import UniversalCritic
from a2c_ppo_acktr.storage import ExtendableStorage

class Heap(object):
    def __init__(self, num_inputs, num_hidden, num_outputs, action_space, post_process=sigmoid, args=None):
        num_units = num_inputs + num_hidden + num_outputs
        assert num_inputs + num_outputs <= num_units
        assert args.discount_mode in ['global', 'parallel', 'sequential'], 'Unknown discount mode: %s' % args.discount_mode
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.threshold = args.threshold
        self.slip_reward = args.slip_reward
        self.signal_split = args.signal_split
        self.postwidth = args.postwidth
        self.bandwidth = args.bandwidth
        self.gamma = args.gamma
        self.num_mini_batch = args.num_mini_batch
        self.distance_edge = args.distance_edge
        self.post_process = post_process
        self.discount_mode = args.discount_mode
        self.critic = UniversalCritic(self.postwidth, self.bandwidth, hidden_size=args.critic_hidden)
        self.activate_map = [0 for _ in range(num_units)]
        self.unit_actions = []
        self.weights = []
        self.build_postcodes()
        self.build_index()
        self.build_units(args, action_space)
        self.memory = ExtendableStorage()
        
    
    def build_postcodes(self):
        self.postcodes = [npr.random(self.postwidth) for _ in range(self.num_units)]
    
    def build_index(self):
        self.index = AnnoyIndex(self.postwidth, 'euclidean')
        for i, postcode in enumerate(self.postcodes[self.num_inputs:]):
            self.index.add_item(i + self.num_inputs, postcode)
        self.index.build(50)
    
    def build_units(self, args, action_space):
        self.units = [Unit(1, 1 + self.postwidth + self.bandwidth, self.postcodes[i], args=args, critic=self.critic) for i in range(self.num_inputs)]
        self.units += [Unit(self.postwidth + self.bandwidth, 1 + self.postwidth + self.bandwidth, self.postcodes[self.num_inputs + i], args=args, critic=self.critic) for i in range(self.num_units - self.num_inputs - self.num_outputs)]
        self.units += [Unit(self.postwidth + self.bandwidth, 2, self.postcodes[self.num_units - self.num_outputs + i], args=args, critic=self.critic, action_space=action_space) for i in range(self.num_outputs)]
    
    def __call__(self, states, n=10):
        assert len(states) == self.num_inputs
        for j, (unit, state) in enumerate(zip(self.units[:self.num_inputs], states)):
            _value, _action, _action_log_prob, _recurrent_hidden_states = unit(np.array([state]), put=False)
            _state = torch.stack((torch.zeros(self.postwidth).float(), torch.Tensor([state]).float().expand(self.bandwidth))).view(-1)
            self.memory.insert(_state, _action[0], _action_log_prob, _value, unit_id=j)
            if self.discount_mode == 'sequential':
                self.memory.discount()
            unit_action = _action[0].numpy()
            activation = sigmoid(unit_action[0])
            if activation > self.threshold:
                postcode = unit_action[1:self.postwidth + 1]
                msg = softmax(unit_action[-self.bandwidth:])
                dest_units, dest_distances = self.index.get_nns_by_vector(postcode, self.signal_split, include_distances=True)
                for dest_unit_id, dest_unit_dist in zip(dest_units, dest_distances):
                    if dest_unit_dist < self.distance_edge:
                        self.unit_actions.append((j, dest_unit_id, msg))
                self.activate_map[j] += 1
            else:
                self.memory.reward(self.slip_reward)
        if self.discount_mode == 'parallel':
            self.memory.discount()
        for k in range(n):
            output_actions = self.__proc()
            ret_actions = [None for _ in range(self.num_outputs)]
            cnts = [0 for _ in range(self.num_outputs)]
            for output_id, output_action in output_actions:
                try:
                    ret_actions[output_id] += output_action
                except:
                    ret_actions[output_id] = output_action
                cnts[output_id] += 1
            if output_actions:
                if self.discount_mode == 'global':
                    self.memory.discount()
                return np.array(ret_actions)/np.array(cnts), k
        if self.discount_mode == 'global':
            self.memory.discount()
        return [], n
    
    def __proc(self):
        next_unit_actions = []
        action = []
        for sender_id, dest_id, msg in self.unit_actions:
            unit = self.units[dest_id]
            sender_postcode = self.postcodes[sender_id]
            dest_state = np.hstack([sender_postcode, msg])
            _value, _action, _action_log_prob, _recurrent_hidden_states = unit(dest_state, put=False)
            _state = torch.from_numpy(dest_state).float().view(-1)
            unit_action = _action[0].numpy()
            if self.num_units - dest_id <= self.num_outputs:
                partial_action = unit_action[0]
                self.memory.insert(_state, _action[0].expand(self.bandwidth + self.postwidth + 1).float(), _action_log_prob, _value, unit_id=dest_id)
                if self.post_process:
                    partial_action = self.post_process(partial_action)
                action.append((self.num_units - dest_id - 1, partial_action))
                self.activate_map[dest_id] += 1
            else:
                self.memory.insert(_state, _action[0], _action_log_prob, _value, unit_id=dest_id)
                activation = sigmoid(unit_action[0])
                if activation > self.threshold:
                    postcode = unit_action[1:self.postwidth + 1]
                    msg = softmax(unit_action[-self.bandwidth:])
                    dest_units, dest_distances = self.index.get_nns_by_vector(postcode, self.signal_split + 1, include_distances=True)
                    if dest_id not in dest_units:
                        dest_units = dest_units[:-1]
                        dest_distances = dest_distances[:-1]
                    for dest_unit_id, dest_dist in zip(dest_units, dest_distances):
                        if dest_id == dest_unit_id:
                            continue
                        if dest_dist < self.distance_edge:
                            next_unit_actions.append((dest_id, dest_unit_id, msg))
                    self.activate_map[dest_id] += 1
                else:
                    self.memory.reward(self.slip_reward)
            if self.discount_mode == 'sequential':
                self.memory.discount()
        if self.discount_mode == 'parallel':
            self.memory.discount()
        self.unit_actions = next_unit_actions
        return action
    
    def reward(self, r):
        """if sum(self.activate_map):
            weights = np.array(self.activate_map)/sum(self.activate_map)
        else:
            weights = np.zeros(self.num_units)
        self.weights.append(weights)
        for weight, unit in zip(weights, self.units):
            unit.reward(weight*r)
        for unit in self.units:
            unit.reward(r)"""
        self.memory.reward(r)
    
    def done(self):
        #for unit in self.units:
        #    unit.done()
        self.memory.done()
        
    def reset(self):
        self.unit_actions = []
        for unit in self.units:
            unit.reset()
    
    def update(self, next_state):
        #for unit in self.units:
        #    if unit.memory.step > 2:
        #        unit.update()
        at_postcode = torch.from_numpy(self.postcodes[0]).float().view(1, self.postwidth)
        from_postcode = torch.zeros((1, self.postwidth)).float()
        msg = torch.from_numpy(next_state[:1]).expand(1, self.bandwidth).float()
        with torch.no_grad():
            next_value = self.critic(at_postcode, from_postcode, msg)
        self.memory.compute_returns(next_value, self.gamma)
        advantages = (self.memory.returns[:-1] - self.memory.value_preds[:-1])
        for i, unit in enumerate(self.units):
            if self.memory.has_enough_for_unit(i):
                unit_data_generator = self.memory.feed_forward_generator(advantages, unit_id=i, num_mini_batch=self.num_mini_batch)
                unit._update(unit_data_generator)
            
    
    def clear_memory(self):
        #for unit in self.units:
        #    unit.clear_memory()
        self.activate_map = [0 for _ in range(self.num_units)]
        self.memory.clear()
    
    def clear_stats(self):
        for unit in self.units:
            unit.clear_stats()
    
    def sleep(self, skip=False):
        if skip:
            self.unit_actions = []
        else:
            while self.unit_actions:
                self.__proc()
    
    def plot_stats(self, n=100):
        def moving_average(a, k=n) :
            ret = np.cumsum(a, dtype=float)
            ret[k:] = ret[k:] - ret[:-k]
            return ret[k - 1:] / k
        print('INPUTS')
        for j in range(self.num_inputs):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(self.weights)))[:1-10*n], moving_average(list(map(lambda x : x[j], self.weights)), 10*n), label=label)
        plt.title('Weight')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[:self.num_inputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.action_losses)))[:1-n], moving_average(u.action_losses), label=label)
        plt.title('Action loss')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[:self.num_inputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.value_losses)))[:1-n], moving_average(u.value_losses), label=label)
        plt.title('Value loss')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[:self.num_inputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.rewards)))[:1-n], moving_average(u.rewards), label=label)
        plt.title('Rewards')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[:self.num_inputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.dist_entropies)))[:1-n], moving_average(u.dist_entropies), label=label)
        plt.title('Dist entropies')
        plt.legend()
        plt.show()
        print('HIDDEN')
        for j in range(self.num_units - self.num_inputs - self.num_outputs):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(self.weights)))[:1-10*n], moving_average(list(map(lambda x : x[self.num_inputs + j], self.weights)), 10*n), label=label)
        plt.title('Weight')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[self.num_inputs:-self.num_outputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.action_losses)))[:1-n], moving_average(u.action_losses), label=label)
        plt.title('Action loss')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[self.num_inputs:-self.num_outputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.value_losses)))[:1-n], moving_average(u.value_losses), label=label)
        plt.title('Value loss')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[self.num_inputs:-self.num_outputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.rewards)))[:1-n], moving_average(u.rewards), label=label)
        plt.title('Rewards')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[self.num_inputs:-self.num_outputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.dist_entropies)))[:1-n], moving_average(u.dist_entropies), label=label)
        plt.title('Dist entropies')
        plt.legend()
        plt.show()
        print('OUTPUTS')
        for j in range(self.num_outputs):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(self.weights)))[:1-10*n], moving_average(list(map(lambda x : x[self.num_units - j - 1], self.weights)), 10*n), label=label)
        plt.title('Weight')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[-self.num_outputs:]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.action_losses)))[:1-n], moving_average(u.action_losses), label=label)
        plt.title('Action loss')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[-self.num_outputs:]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.value_losses)))[:1-n], moving_average(u.value_losses), label=label)
        plt.title('Value loss')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[-self.num_outputs:]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.rewards)))[:1-n], moving_average(u.rewards), label=label)
        plt.title('Rewards')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[-self.num_outputs:]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.dist_entropies)))[:1-n], moving_average(u.dist_entropies), label=label)
        plt.title('Dist entropies')
        plt.legend()
        plt.show()