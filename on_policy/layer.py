from scipy.special import softmax
import numpy as np

from unit import Unit

class Layer(object):
    def __init__(self, num_units, num_inputs, num_outputs, post_width=0, post_process=softmax, args=None):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.post_width = post_width
        self.units = [Unit(self.num_inputs,
                           self.post_width + self.num_outputs,
                           args=args) for _ in range(self.num_units)]
        self.activate_map = [False for _ in range(self.num_units)]
        self.post_process = post_process
    
    def __call__(self, states):
        unit_actions = []
        for j, (unit, state) in enumerate(zip(self.units, states)):
            try:
                assert len(state) == self.num_inputs
                unit_actions.append(unit(state))
                if self.post_width:
                    postcode = np.argmax(unit_actions[-1][:self.post_width])
                    msg = unit_actions[-1][self.post_width:]
                    unit_actions[-1] = (postcode, msg)
                self.activate_map[j] = True
            except:
                unit_actions.append(None)
                self.activate_map[j] = False
        if self.post_width:
            outputs = [None for _ in range(self.post_width)]
            for unit_action in unit_actions:
                try:
                    postcode, msg = unit_action
                except:
                    continue
                try:
                    outputs[postcode] += msg
                except:
                    outputs[postcode] = msg
            if self.post_process:
                for i, output in enumerate(outputs):
                    try:
                        outputs[i] = self.post_process(output)
                    except:
                        continue
            return outputs
        else:
            if self.post_process:
                for i, action in enumerate(unit_actions):
                    try:
                        unit_actions[i] = self.post_process(action)
                    except:
                        continue
            return unit_actions
    
    def reward(self, reward):
        for flag, unit in zip(self.activate_map, self.units):
            if flag:
                unit.reward(reward)
    
    def done(self):
        for unit in self.units:
            unit.done()
    
    def update(self):
        for unit in self.units:
            unit.update()
    
    def clear_memory(self):
        for unit in self.units:
            unit.clear_memory()
    
    def clear_stats(self):
        for unit in self.units:
            unit.clear_stats()