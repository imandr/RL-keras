import random

class ReplayMemory(object):
    
    def __init__(self, generators, low_water_mark, high_water_mark=None):
        self.Memory = []
        self.Generators = generators        # generator and sample sizes
        self.LowWater = low_water_mark
        self.HighWater = high_water_mark or int(low_water_mark * 1.2)
        self.RefreshRate = 0.1
        self.I = 0
        self.NGenerated = [0]*len(self.Generators)

    def fill(self, n):
        if len(self.Memory) < max(self.LowWater, n):
            while len(self.Memory) < max(self.HighWater, n):
                for ig, (g, ng) in enumerate(self.Generators):
                    sample = g.generate(ng)
                    self.Memory += sample
                    self.NGenerated[i] += len(sample)
            random.shuffle(self.Memory)    
            self.I = 0      

    def sample(self, n):
        self.fill(n)
        s = self.Memory[self.I:self.I+n]
        push_out = int(n*self.RefreshRate)
        self.Memory = self.Memory[push_out:]
        self.I += n - push_out
        return s
