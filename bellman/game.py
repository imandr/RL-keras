import numpy as np, random
from RLKeras import ReplayMemory

class Game(object):
    
    Delta = 0.1
    NActions = 4
    StateDim = 2

    Moves = np.array(
        [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0)
        ]
    ) * Delta

    def init(self, n):
        # returns ranodm states, done
        states = self.randomStates(n)
        return states       
    
    def step(self, states, actions):
        n = len(states)
        states1 = states.copy()
        done = np.zeros((n,), dtype=np.bool)
        rewards = np.zeros((n,))
        for i, (state, action) in enumerate(zip(states, actions)):
            states1[i,:] += self.Moves[action]
            x, y = states1[i,:]
            final = x > 1.0 or y > 1.0
            if final:
                z = x if y > 1.0 else y
                r = 1-2*z
                done[i] = True
                rewards[i] = r
        return states1, rewards, done
        
    def randomStates(self, n):
        return np.random.random((n,2))
        
    def randomActions(self, n):
        return np.random.randint(0, self.NActions, n)
        
                
class Generator(object):
    
    def __init__(self, game):
        self.Game = game
        self.NActions = self.Game.NActions
    
    def __iter__(self):
        return self.generate()
    

class RandomGenerator(Generator):
    
    def __init__(self, game, mbsize):
        Generator.__init__(self, game)
        self.MBSize = mbsize
        
    def generate(self):
        mbsize = self.MBSize
        while True:
            s0 = self.Game.randomStates(mbsize)
            direction = self.Game.randomActions(mbsize)
            #print direction
            s1, rewards, final = self.Game.step(s0, direction)
            yield [s0, direction, s1, final], rewards[:,None]
        
    
class GameGenerator(Generator):
    
    
    def __init__(self, game, qmodel, mbsize, temperature):
        Generator.__init__(self, game)
        self.MBSize = mbsize
        self.QModel = qmodel
        self.Temp = temperature
        self.Memory = []
        self.TMax = 20
        
    def play_games(self, ngames):
        s = self.Game.randomStates(ngames)
        done = np.zeros((ngames,), dtype=np.bool)
        record = []
        #all_actions = np.arange(self.NActions)
        #print all_actions
        t = self.TMax
        while len(s) > 0 and t > 0:
            q = self.QModel.predict_on_batch(s)
            q -= np.max(q, axis=-1, keepdims=True)
            exp = np.exp(q/self.Temp)
            probs = exp / np.sum(exp, axis=-1, keepdims=True)
            actions = [np.random.choice(self.Game.NActions, 1, p=p)[0] for p in probs]
            s1, rewards, done = self.Game.step(s, actions)
            record += list(zip(s, actions, s1, rewards, done))
            #print s1.shape, done.shape
            s = (s1[done==False]).copy()
            t -= 1
            #print len(s)
        return record
            
            

    def fill_memory(self):
        while len(self.Memory) < self.MBSize * 10:
            record = self.play_games(self.MBSize)
            self.Memory += record
        random.shuffle(self.Memory)            
        
    def generate(self):
        
        i = 0
        
        while True:
            if len(self.Memory) < self.MBSize:
                self.fill_memory()
            segment = self.Memory[:self.MBSize]
            self.Memory = self.Memory[self.MBSize:]
            s0, action, s1, reward, final = zip(*segment)
            yield [np.array(s0), np.array(action), np.array(s1), np.array(final)], np.array(reward)[:,None]
            
                
        
        
    

class GameGenerator1(Generator):
    
    
    def __init__(self, game, qmodel, mbsize, temperature):
        Generator.__init__(self, game)
        self.MBSize = mbsize
        self.QModel = qmodel
        self.Temp = temperature
        self.MSize = mbsize*10
        self.Memory = ReplayMemory(self.MSize)
        
    def play_games(self, ngames):
        s = self.Game.randomStates(ngames)
        done = np.zeros((ngames,), dtype=np.bool)
        record = []
        #all_actions = np.arange(self.NActions)
        #print all_actions
        while len(s) > 0:
            q = self.QModel.predict_on_batch(s)
            q -= np.max(q, axis=-1, keepdims=True)
            exp = np.exp(q/self.Temp)
            probs = exp / np.sum(exp, axis=-1, keepdims=True)
            actions = [np.random.choice(self.Game.NActions, 1, p=p)[0] for p in probs]
            s1, rewards, done = self.Game.step(s, actions)
            record += list(zip(s, actions, s1, rewards, done))
            #print s1.shape, done.shape
            s = (s1[done==False]).copy()
            #print len(s)
        for tup in record:
            self.Memory.add(tup, 1.0)
            
    def generate(self):
        
        i = 0
        
        while True:
            self.play_games(self.MBSize)
            while self.Memory.size() < self.MBSize:
                self.play_games(2)
            segment = self.Memory.sample(self.MBSize)
            s0, action, s1, reward, final = zip(*segment)
            yield [np.array(s0), np.array(action), np.array(s1), np.array(final)], np.array(reward)[:,None]
            
                
        
        
    

