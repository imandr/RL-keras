import random

class ReplayMemory:
    
    M = 1
    
    def __init__(self, size):
        self.MaxSize = size
        self.HighWater = int(size*1.3)
        self.Memory = []
        self.ShortTermMemory = []
        self.Age = 0
        
    def add(self, tup):   
        self.ShortTermMemory.append(tup)
        self.Age += 1
            
    def add_to_long(self, tups):
        self.Memory.extend(tups)
        if len(self.Memory) > self.HighWater:
            self.Memory = random.sample(self.Memory, self.MaxSize)
        
    def sample(self, size):
        shorts = self.ShortTermMemory[:size]
        n_short = len(shorts)
        self.ShortTermMemory = self.ShortTermMemory[n_short:]
        n_long = min(len(self.Memory), size - n_short)
        #print "n_short/n_long:", n_short, n_long
        longs = random.sample(self.Memory, n_long) if n_long > 0 else []
        self.add_to_long(shorts)
        return shorts + longs
        
    def size(self):
        return len(self.Memory)+len(self.ShortTermMemory)

    def sizes(self):
        return len(self.Memory), len(self.ShortTermMemory)
