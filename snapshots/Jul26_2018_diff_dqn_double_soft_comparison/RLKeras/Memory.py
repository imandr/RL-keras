import random, math
import numpy as np

class ReplayMemory:
    
    M = 1
    
    def __init__(self, size, v_selectivity = True, bypass_short_term = True):
        self.MaxSize = size
        self.HighWater = int(size*1.3)
        self.Memory = []
        self.ShortTermMemory = []
        self.Age = 0
        self.Known = set()
        self.MeanSQ = 0.0
        self.MeanW = 0.0
        self.Alpha = 0.01
        self.VSel = v_selectivity
        self.BypassShortTerm = bypass_short_term
        self.ISample = 0
        
    def makeHashable(self, tup):
        return tuple([x if not isinstance(x, (list, np.ndarray)) else tuple(x) for x in tup])
        
    def add(self, tup, weight):
        self.MeanSQ += self.Alpha*(weight**2-self.MeanSQ)
        self.MeanW += self.Alpha*(weight-self.MeanW)
        sigma = self.MeanSQ - self.MeanW**2
        #if sigma == 0.0 or math.exp(-(weight-self.MeanW)**2/(2*sigma)) < random.random():
        if not self.VSel or sigma == 0.0 or math.exp(-abs(weight-self.MeanW)/sigma) < random.random():
            key = self.makeHashable(tup)
            #print key
            if not key in self.Known:
                if self.BypassShortTerm:
                    self.add_to_long([tup])
                else:
                    self.ShortTermMemory.append(tup)
                #print tup
                self.Known.add(key)
                self.Age += 1
            #else:
            #    print "repeated"
            
    def add_to_long(self, tups):
        self.Memory.extend(tups)
        if len(self.Memory) > self.HighWater:
            self.Memory = random.sample(self.Memory, self.MaxSize)
            self.Known = set(map(self.makeHashable, self.Memory+self.ShortTermMemory))
            self.ISample = self.ISample % len(self.Memory)
        
    def sample(self, size):
        shorts = self.ShortTermMemory[:size]
        n_short = len(shorts)
        self.ShortTermMemory = self.ShortTermMemory[n_short:]
        
        N = len(self.Memory)
        n_need = min(size-n_short, N)
        longs = []
        if n_need > 0:
            if n_need >= N:
                longs = self.Memory
            else:
                longs = self.Memory[self.ISample:self.ISample+n_need]
                n_need -= len(longs)
                if n_need > 0:
                    longs += self.Memory[:n_need]
                    random.shuffle(self.Memory)
                self.ISample = (self.ISample + len(longs)) % N
        self.add_to_long(shorts)
        #print "Memory: ISample:", self.ISample, "   size:",N
        return shorts + longs
        
    def size(self):
        return len(self.Memory)+len(self.ShortTermMemory)

    def sizes(self):
        return len(self.Memory), len(self.ShortTermMemory)
