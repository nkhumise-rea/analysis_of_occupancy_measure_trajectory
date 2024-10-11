import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', 
                        ('state','next_state','action','reward', 'done')
                        )
                        
class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, done):
        self.memory.append(Transition(
            state, next_state, action, reward, done)
            )    
        
    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch
    
    def __len__(self):
        return len(self.memory)