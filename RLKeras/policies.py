import random
import numpy as np

class GreedyEpsPolicy:

    def __init__(self, epsilon = 0.0):
        self.Epsilon = epsilon
    
    def choose(self, qs, valids):
        # valids is list of actions
        if random.random() < self.Epsilon:
            return random.choice(valids)
        else:
            return valids[np.argmax(qs[valids])]
            
    __call__ = choose
    
