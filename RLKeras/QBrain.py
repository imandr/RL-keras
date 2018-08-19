from .Memory import ReplayMemory
import numpy as np
from .tools import format_batch

class QBrain:    
    def __init__(self, model, kind = "dqn", gamma = 0.99, 
                    # memory parameters
                    bypass_short_term = True, memory = None, memory_size = 200000, v_selectivity = False,
                    # dual DQN only
                    diff_qnet_weight = 0.99,
                    qnet_soft_update = None,             
                    qnet_hard_update = 10000             # train samples between live->target network copies
                    ):
        if kind == "diff":
            from .experimental import DifferentialQNet
            self.QNet = DifferentialQNet(model, gamma = gamma)
        elif kind == "diff2":
            from .experimental import DifferentialQNet2
            self.QNet = DifferentialQNet2(model, gamma = gamma, rel_weight=diff_qnet_weight)
        elif kind == "qv":
            from .experimental import QVNet
            self.QNet = QVNet(model, gamma = gamma, rel_weight=diff_qnet_weight)
        elif kind in ("dqn","double","naive"):
            from .QNet import DQN
            self.QNet = DQN(model, kind=kind, gamma=gamma, soft_update = qnet_soft_update,
                    hard_update_samples=qnet_hard_update)
        else:
            raise ValueError("Unknown QNet kind %s" % (kind,))
        self.Memory = memory or ReplayMemory(memory_size, v_selectivity=v_selectivity, bypass_short_term=bypass_short_term)
    
    @property
    def trainSamples(self):
        return self.QNet.TrainSamples
        
    @property
    def age(self):
        return self.Memory.Age
        
    def memorize(self, tup, w):
        #print "memorize(%s)" % (tup,)
        self.Memory.add(tup, w)

    def train(self, sample_size, batch_size):
        sample = self.Memory.sample(sample_size)
        #print "QBrain: got sample:", len(sample), batch_size
        return self.QNet.train(sample, batch_size)
        
    def get_weights(self):
        return self.QNet.get_weights()
        
    def blend_weights(self, alpha, weights):
        self.QNet.blend_weights(alpha, weights)
        
    def set_weights(self, weights):
        self.QNet.set_weights(weights)
        
    def blend(self, alpha, other):
        w = other.get_weights()
        self.blend_weights(alpha, w)
        
    def transfer(self, other):
        self.set_weights(other.get_weights())
        
    def recordSize(self):
        return self.Memory.size()
        
    def compile(self, optimizer, metrics):
        self.QNet.compile(optimizer, metrics)
        
    def qvector(self, observation):
        x = format_batch([observation])
        #print "qvector: x=", x
        qv = self.QNet.compute(x)[0]
        #print "qv:", qv
        return qv

    
    