from .Memory import ReplayMemory
from .QNet import DifferentialQNet, DualQNet
import numpy as np
from .tools import format_batch

class QBrain:    
    def __init__(self, model, typ = "diff", gamma = 0.99, soft_update = None, memory = None, memory_size = 100000,
                    v_selectivity = True, bypass_short_term = False):
        if typ == "diff":
            self.QNet = DifferentialQNet(model, gamma = gamma)
        elif typ == "dual":
            self.QNet = DualQNet(model, gamma=gamma, soft_update=soft_update)
        else:
            raise ValueError("Unknown QNet type %s" % (typ,))
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
        #print len(sample), batch_size
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
        
    def update(self):
        self.QNet.update()
        
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

    
    