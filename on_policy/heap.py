import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax, expit as sigmoid
import numpy as np
import numpy.random as npr
from annoy import AnnoyIndex

from unit import Unit

class Heap(object):
    def __init__(self, num_units, num_inputs, num_outputs, post_process=sigmoid, args=None):
        assert num_inputs + num_outputs <= num_units
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.threshold = args.threshold
        self.slip_reward = args.slip_reward
        self.signal_split = args.signal_split
        self.postwidth = args.postwidth
        self.bandwidth = args.bandwidth
        self.units = [Unit(1, 1 + args.postwidth + args.bandwidth, args=args) for _ in range(num_inputs)]
        self.units += [Unit(args.postwidth + args.bandwidth, 1 + args.postwidth + args.bandwidth, args=args) for _ in range(num_units - num_inputs - num_outputs)]
        self.units += [Unit(args.postwidth + args.bandwidth, 2, args=args) for _ in range(num_outputs)]
        self.postcodes = [npr.random(args.bandwidth) for _ in range(num_inputs)]
        self.postcodes += [npr.random(args.bandwidth) for _ in range(num_units - num_inputs)]
        self.index = AnnoyIndex(args.bandwidth, 'euclidean')
        for i, postcode in enumerate(self.postcodes[num_inputs:]):
            self.index.add_item(i + num_inputs, postcode)
        self.index.build(50)
        self.post_process = post_process
        self.activate_map = [0 for _ in range(num_units)]
        self.unit_actions = []
    
    def __call__(self, states, n=10):
        assert len(states) == self.num_inputs
        for j, (unit, state) in enumerate(zip(self.units[:self.num_inputs], states)):
            unit_action = unit(np.array([state]))
            activation = sigmoid(unit_action[0])
            if activation > self.threshold:
                postcode = unit_action[1:self.postwidth + 1]
                msg = softmax(unit_action[-self.bandwidth:])
                dest_units = self.index.get_nns_by_vector(postcode, self.signal_split)
                for dest_unit_id in dest_units:
                    self.unit_actions.append((j, dest_unit_id, msg))
                self.activate_map[j] += 1
                unit.reward(self.slip_reward)
        action = np.zeros(self.num_outputs)
        counts = np.zeros(self.num_outputs)
        for _ in range(n):
            output_actions = self.__proc()
            for output_id, output_action in output_actions:
                action[output_id] += output_action
                counts[output_id] += 1
        zeros_mask = np.where(counts == 0)
        counts[zeros_mask] = 1
        return action/counts, zeros_mask
    
    def __proc(self):
        next_unit_actions = []
        action = []
        for sender_id, dest_id, msg in self.unit_actions:
            unit = self.units[dest_id]
            sender_postcode = self.postcodes[sender_id]
            dest_state = np.hstack([sender_postcode, msg])
            unit_action = unit(dest_state)
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
                    dest_units = self.index.get_nns_by_vector(postcode, self.signal_split)
                    for dest_unit_id in dest_units:
                        next_unit_actions.append((dest_id, dest_unit_id, msg))
                    self.activate_map[dest_id] += 1
                    unit.reward(self.slip_reward)
        self.unit_actions = next_unit_actions
        return action
    
    def reward(self, r):
        weights = softmax(self.activate_map)
        for weight, unit in zip(weights, self.units):
            unit.reward(weight*r)
    
    def done(self):
        for unit in self.units:
            unit.done()
    
    def update(self):
        for unit in self.units:
            if unit.memory.step > 2:
                unit.update()
    
    def clear_memory(self):
        for unit in self.units:
            unit.clear_memory()
    
    def clear_stats(self):
        for unit in self.units:
            unit.clear_stats()
    
    def plot_stats(self, n=100):
        def moving_average(a) :
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
        for j, u in enumerate(self.units):
            label = 'Hidden %d' % (j + 1 - self.num_inputs)
            if j <= self.num_inputs:
                label = 'Input %d' % (j + 1)
            elif self.num_units - j <= self.num_outputs:
                label = 'Output %d' % (self.num_units - j)
            plt.plot(list(range(len(u.action_losses)))[:1-n], moving_average(u.action_losses), label=label)
        plt.title('Action loss')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units):
            label = 'Hidden %d' % (j + 1 - self.num_inputs)
            if j <= self.num_inputs:
                label = 'Input %d' % (j + 1)
            elif self.num_units - j <= self.num_outputs:
                label = 'Output %d' % (self.num_units - j)
            plt.plot(list(range(len(u.value_losses)))[:1-n], moving_average(u.value_losses), label=label)
        plt.title('Value loss')
        plt.legend()
        plt.show()
        for j, u in enumerate(self.units):
            label = 'Hidden %d' % (j + 1 - self.num_inputs)
            if j <= self.num_inputs:
                label = 'Input %d' % (j + 1)
            elif self.num_units - j <= self.num_outputs:
                label = 'Output %d' % (self.num_units - j)
            plt.plot(list(range(len(u.dist_entropies)))[:1-n], moving_average(u.dist_entropies), label=label)
        plt.title('Dist entropies')
        plt.legend()
        plt.show()