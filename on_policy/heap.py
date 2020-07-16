import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax, expit as sigmoid
import numpy as np
import numpy.random as npr
from annoy import AnnoyIndex
import os
import torch
from tqdm import tqdm
from itertools import chain

from unit import Unit
from a2c_ppo_acktr.storage import ExtendableTransformerStorage

def update_unit(i, unit, memories, advantageses, pbar, postwidth=0):
    data_generators = []
    for memory, advantages in zip(memories, advantageses):
        if memory.has_enough_for(i):
            unit_data_generator = memory.feed_forward_generator(advantages, unit_id=i)
            data_generators.append(unit_data_generator)
    if data_generators:
        data_generator = chain(*data_generators)
        unit.update(data_generator, postwidth=postwidth)
    pbar.update(1)

class Heap(object):
    def __init__(self, num_inputs, num_hidden, num_outputs, post_process=sigmoid, args=None):
        num_units = num_inputs + num_hidden + num_outputs
        assert args.discount_mode in ['global', 'parallel', 'sequential'], 'Unknown discount mode: %s' % args.discount_mode
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.quantile_truncate = args.quantile_truncate
        self.threshold = args.threshold
        self.signal_split = args.signal_split
        self.postwidth = args.postwidth
        self.bandwidth = args.bandwidth
        self.gamma = args.gamma
        self.post_process = post_process
        self.discount_mode = args.discount_mode
        self.args = args
        self.unit_actions_map = {}
        self.build_postcodes()
        self.build_index()
        self.build_units()
        self.memory_map = {}
    
    @classmethod
    def load(cls, fname):
        heap_dict = torch.load(fname)
        heap = cls(1, 1, 1, args=heap_dict['args'])
        heap.args = heap_dict['args']
        heap.num_units = heap_dict['num_units']
        heap.num_inputs = heap_dict['num_inputs']
        heap.num_outputs = heap_dict['num_outputs']
        heap.num_hidden = heap_dict['num_hidden']
        heap.quantile_truncate = heap_dict['quantile_truncate']
        heap.threshold = heap_dict['threshold']
        heap.signal_split = heap_dict['signal_split']
        heap.postwidth = heap_dict['postwidth']
        heap.bandwidth = heap_dict['bandwidth']
        heap.gamma = heap_dict['gamma']
        heap.post_process = heap_dict['post_process']
        heap.discount_mode = heap_dict['discount_mode']
        heap.postcodes = heap_dict['postcodes']
        heap.build_index()
        heap.units = []
        for unit_dict in heap_dict['units']:
            num_inputs = unit_dict['num_inputs']
            num_outputs = unit_dict['num_outputs']
            action_space = unit_dict['action_space']
            signal_split = heap_dict['signal_split']
            unit = Unit(num_inputs, num_outputs, args=heap.args, is_input=is_input, is_output=is_output)
            unit.actor_critic.load_state_dict(unit_dict['actor_critic_state'])
            heap.units.append(unit)
        heap.unit_actions_map = {}
        heap.memory_map = {}
        return heap
    
    def save(self, fname='heap.pth'):
        heap_dict = {
            'num_units' : self.num_units,
            'num_inputs' : self.num_inputs,
            'num_outputs' : self.num_outputs,
            'num_hidden' : self.num_hidden,
            'quantile_truncate' : self.quantile_truncate,
            'threshold' : self.threshold,
            'signal_split' : self.signal_split,
            'postwidth' : self.postwidth,
            'bandwidth' : self.bandwidth,
            'gamma' : self.gamma,
            'post_process' : self.post_process,
            'discount_mode' : self.discount_mode,
            'postcodes' : self.postcodes,
            'units' : [],
            'args' : self.args
        }
        for unit in self.units:
            unit_dict = {
                'num_inputs' : unit.num_inputs,
                'num_outputs' : unit.num_outputs,
                'action_space' : unit.action_space,
                'actor_critic_state' : unit.actor_critic.state_dict()
            }
            heap_dict['units'].append(unit_dict)
        torch.save(heap_dict, fname)
    
    def build_postcodes(self):
        self.postcodes = [npr.random(self.postwidth) for _ in range(self.num_units)]
    
    def build_index(self):
        self.index = AnnoyIndex(self.postwidth, 'euclidean')
        for i, postcode in enumerate(self.postcodes[self.num_inputs:]):
            self.index.add_item(i + self.num_inputs, postcode)
        self.index.build(50)
    
    def build_units(self):
        self.units = [Unit(1, 1 + self.postwidth + self.bandwidth, args=self.args, is_input=True) for i in range(self.num_inputs)]
        self.units += [Unit(1 + self.postwidth + self.bandwidth, 1 + self.postwidth + self.bandwidth, args=self.args) for i in range(self.num_units - self.num_inputs - self.num_outputs)]
        self.units += [Unit(1 + self.postwidth + self.bandwidth, 2, args=self.args, is_output=True) for i in range(self.num_outputs)]
    
    def get_inner_state(self, env=''):
        if env not in self.unit_actions_map:
            self.unit_actions_map[env] = []
            self.memory_map[env] = ExtendableTransformerStorage()
        unit_actions = self.unit_actions_map[env]
        memory = self.memory_map[env]
        return unit_actions, memory
    
    def __call__(self, states, n=10, trace_value=False, env=''):
        unit_actions, memory = self.get_inner_state(env)
        if trace_value:
            value_s = []
        assert len(states) == self.num_inputs
        for j, (unit, state) in enumerate(zip(self.units[:self.num_inputs], states)):
            _value, _action, _action_log_prob = unit(np.array([state]))
            _state = torch.Tensor([state]).float()
            if trace_value:
                value_s.append(_value.item())
            else:
                memory.insert(_state, _action, _action_log_prob, _value, unit_id=j)
                if self.discount_mode == 'sequential':
                    memory.discount()
            for a_id in range(self.signal_split):
                unit_action = _action[a_id].numpy()
                activation = sigmoid(unit_action[0])
                if activation > self.threshold:
                    postcode = unit_action[1:self.postwidth + 1]
                    msg = softmax(unit_action[-self.bandwidth:])
                    dest_unit_id = self.index.get_nns_by_vector(postcode, 1)[0]
                    unit_actions.append((j, dest_unit_id, activation, msg))
                    if not trace_value:
                        memory.add_target_pc(torch.Tensor(self.postcodes[dest_unit_id]).float())
                elif not trace_value:
                    memory.add_target_pc(None)
        if self.discount_mode == 'parallel' and not trace_value:
            memory.discount()
        for k in range(n):
            if trace_value:
                traced_value_s, flag = self.__proc(trace_value=True, env=env)
                value_s += traced_value_s
                if flag:
                    value_s.sort()
                    q_low = int(len(value_s)*self.quantile_truncate)
                    q_high = int(len(value_s)*(1 - self.quantile_truncate))
                    return sum(value_s[q_low:q_high])/(q_high - q_low)
            else:
                output_actions = self.__proc(env=env)
                ret_actions = [None for _ in range(self.num_outputs)]
                for output_id, output_action in output_actions:
                    ret_actions[output_id] = output_action
                if output_actions:
                    if self.discount_mode == 'global':
                        memory.discount()
                    return ret_actions, k
        if trace_value:
            value_s.sort()
            q_low = int(len(value_s)*self.quantile_truncate)
            q_high = int(len(value_s)*(1 - self.quantile_truncate))
            return sum(value_s[q_low:q_high])/(q_high - q_low)
        else:
            if self.discount_mode == 'global':
                memory.discount()
            return [], n
    
    def __proc(self, trace_value=False, env=''):
        _, memory = self.get_inner_state(env)
        if trace_value:
            value_s = []
        next_unit_actions = []
        action = []
        unit_states = self._prepare(env=env)
        for dest_id, dest_state in unit_states:
            unit = self.units[dest_id]
            _value, _action, _action_log_prob = unit(dest_state)
            _state = torch.from_numpy(dest_state).float()
            if trace_value:
                value_s.append(_value.item())
            else:
                memory.insert(_state, _action, _action_log_prob, _value, unit_id=dest_id)
                if self.discount_mode == 'sequential':
                    memory.discount()
            for a_id in range(self.signal_split):
                if a_id < _action.shape[0]:
                    unit_action = _action[a_id].numpy()
                    activation = sigmoid(unit_action[0])
                    if activation > self.threshold:
                        if self.num_units - dest_id <= self.num_outputs:
                            partial_action = unit_action[1]
                            if self.post_process:
                                partial_action = self.post_process(partial_action)
                            action.append((self.num_units - dest_id - 1, partial_action))
                        else:
                            postcode = unit_action[1:self.postwidth + 1]
                            msg = softmax(unit_action[-self.bandwidth:])
                            dest_units = self.index.get_nns_by_vector(postcode, 2)
                            dest_unit_id = dest_units[0]
                            if dest_unit_id == dest_id:
                                dest_unit_id = dest_units[1]
                            next_unit_actions.append((dest_id, dest_unit_id, activation, msg))
                            if not trace_value:
                                memory.add_target_pc(torch.Tensor(self.postcodes[dest_unit_id]).float())
                    elif not trace_value:
                        memory.add_target_pc(None)
        self.unit_actions_map[env] = next_unit_actions
        if trace_value:
            return value_s, bool(action)
        else:
            if self.discount_mode == 'parallel':
                memory.discount()
            return action
    
    def _prepare(self, env=''):
        unit_actions, _ = self.get_inner_state(env)
        mapped_unit_actions = {}
        for sender_id, dest_id, sender_activation, msg in unit_actions:
            sender_pc = self.postcodes[sender_id]
            try:
                mapped_unit_actions[dest_id]['pcs'] += [sender_pc]
                mapped_unit_actions[dest_id]['activations'] += [sender_activation]
                mapped_unit_actions[dest_id]['msgs'] += [msg]
            except:
                mapped_unit_actions[dest_id] = {
                                                    'pcs' : [sender_pc],
                                                    'activations' : [sender_activation],
                                                    'msgs' : [msg]
                                               }
                
        unit_states = []
        for dest_id in mapped_unit_actions:
            sender_pcs = np.vstack(mapped_unit_actions[dest_id]['pcs'])
            sender_acts = np.vstack(mapped_unit_actions[dest_id]['activations'])
            sender_msgs = np.vstack(mapped_unit_actions[dest_id]['msgs'])
            dest_state = np.hstack((sender_pcs, sender_acts, sender_msgs))
            unit_states.append((dest_id, dest_state))
        return unit_states
    
    def reward(self, r, env=''):
        assert r >= -1 and r <= 1, 'Provided reward is out of range [-1, 1]'
        r *= 1 - self.gamma
        _, memory = self.get_inner_state(env)
        memory.reward(r)
    
    def done(self, env=''):
        _, memory = self.get_inner_state(env)
        memory.done()
        
    def reset(self, env=''):
        _, _ = self.get_inner_state(env)
        self.unit_actions_map[env] = []
    
    def update(self, next_states_dict):
        memories = []
        advantageses = []
        for env in next_states_dict:
            next_state = next_states_dict[env]
            next_value = self(next_state, trace_value=True, env=env)
            _, memory = self.get_inner_state(env)
            memory.compute_returns(torch.tensor([next_value]).float(), self.gamma)
            advantages = (memory.returns[:-1] - memory.value_preds[:-1])
            memories.append(memory)
            advantageses.append(advantages)
        with tqdm(total=self.num_units, desc="Updating") as pbar:
            for i, unit in enumerate(self.units):
                update_unit(i, unit, memories, advantageses, pbar, postwidth=self.postwidth)            
    
    def clear_memory(self, env=None):
        if env is not None:
            _, memory = self.get_inner_state(env)
            memory.clear()
        else:
            for env in self.memory_map:
                self.memory_map[env].clear()
    
    def clear_stats(self):
        for unit in self.units:
            unit.clear_stats()
    
    def plot_stats(self, n=100):
        def moving_average(a, k=n) :
            ret = np.cumsum(a, dtype=float)
            ret[k:] = ret[k:] - ret[:-k]
            return ret[k - 1:] / k
        print('INPUTS')
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
            plt.plot(list(range(len(u.dist_entropies)))[:1-n], moving_average(u.dist_entropies), label=label)
        plt.title('Dist entropies')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[:self.num_inputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.direction_losses)))[:1-n], moving_average(u.direction_losses), label=label)
        plt.title('Direction losses')
        plt.legend()
        plt.show()
        print('HIDDEN')
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
            plt.plot(list(range(len(u.dist_entropies)))[:1-n], moving_average(u.dist_entropies), label=label)
        plt.title('Dist entropies')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units[self.num_inputs:-self.num_outputs]):
            label = 'Unit %d' % (j + 1)
            plt.plot(list(range(len(u.direction_losses)))[:1-n], moving_average(u.direction_losses), label=label)
        plt.title('Direction losses')
        plt.legend()
        plt.show()
        print('OUTPUTS')
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
            plt.plot(list(range(len(u.dist_entropies)))[:1-n], moving_average(u.dist_entropies), label=label)
        plt.title('Dist entropies')
        plt.legend()
        plt.show()