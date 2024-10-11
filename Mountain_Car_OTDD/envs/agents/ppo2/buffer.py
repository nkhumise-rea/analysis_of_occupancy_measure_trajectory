import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', 
                        ('state','action','log_pi','reward', 'done', 'value')
                        )
                        
class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, log_pi, reward, done, value):
        self.memory.append(Transition(
            state, action, log_pi, reward, done, value)
            )    
        
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    def issue(self):
        transitions = self.memory #random.sample(self.memory, 64) #
        dataset = Transition(*zip(*transitions))
        dataset_size = len(transitions)
        return dataset,dataset_size
    
    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory.clear()