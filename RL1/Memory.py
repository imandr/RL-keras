import random

class ReplayMemory(object):
    
    def __init__(self, source, low_water_mark, high_water_mark=None):
        self.Memory = []
        self.Source = source        
        self.LowWater = low_water_mark
        self.HighWater = high_water_mark or int(low_water_mark * 2)
        self.RefreshRate = 0.1
        self.Cursor = 0

    def fill(self, n):
        if len(self.Memory) - self.Cursor < n:
            nflush = int(self.RefreshRate * self.Cursor)
            self.Memory = self.Memory[nflush:]
            while len(self.Memory) < max(self.LowWater, n):
                sample = self.Source.generate()
                self.Memory += sample
            random.shuffle(self.Memory)    
            self.Cursor = 0      

    def addSamples(self, samples):
        assert isinstance(sample, list)
        self.Memory = self.Memory + samples
        if len(self.Memory) > self.HighWater:
            self.Memory = self.Memory[-self.HighWater:]
            random.shuffle(self.Memory)
            self.Cursor = 0
            
    def sample(self, n):
        self.fill(n)
        s = self.Memory[self.Cursor:self.Cursor+n]
        self.Cursor += n
        return s
        
    def generate_samples(self, mbsize):
        while True:
            sample = self.sample(mbsize)
            columns = zip(*sample)
            s0 = np.array(columns[0])
            a = np.array(columns[1])
            s1 = np.array(columns[2])
            r = np.array(columns[3])
            f = np.array(columns[4])
            yield s0, a, s1, f, r
            
