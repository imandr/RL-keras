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
            #return np.argmax(qs)
            return valids[np.argmax(qs[valids])]
            
    def __str__(self):
        return "GreedyEpsPolicy(%f)" % (self.Epsilon,)
            
    __call__ = choose
    
class BoltzmannQPolicy:
    """Implement the Boltzmann Q Policy

    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, tau=1., clip=(-500., 500.)):
        self.tau = tau
        self.clip = clip

    def __str__(self):
        return "BoltzmannQPolicy(%f)" % (self.tau,)

    def select_action(self, q_values, valids):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        if self.tau <= 0.0:
            return valids[np.argmax(q_values[valids])]
        q_values = q_values.astype('float64')[valids]
        q_values -= np.max(q_values)

        exp_values = np.exp(q_values / self.tau)
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(valids, p=probs)
        return action
        
    __call__ = select_action
    
