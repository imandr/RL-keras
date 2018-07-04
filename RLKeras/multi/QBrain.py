from ..Memory import ReplayMemory
from ..QNet import QNet
import numpy as np
from ..tools import format_batch

class QBrain:    
    def __init__(self, model, gamma = 0.99, soft_update = None, memory_size = 100000):
        self.QNet = QNet(model, soft_update = soft_update)
        self.Memory = ReplayMemory(memory_size)
        self.Gamma = gamma
    
    @property
    def trainSamples(self):
        return self.QNet.TrainSamples
        
    @property
    def age(self):
        return self.Memory.Age
        
    def memorize(self, tup):
        #print "memorize(%s)" % (tup,)
        self.Memory.add(tup)

    def train(self, sample_size):
        samples = self.Memory.sample(sample_size)
        return self.QNet.train(samples, self.Gamma)
        
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

    
    