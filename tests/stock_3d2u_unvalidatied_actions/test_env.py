import random, math
import numpy as np

class Stock:
    
    TWindow = 5
    
    def __init__(self, price, volatility, noise, tmax):
        self.P0 = price
        self.V = volatility/math.sqrt(float(tmax))
        self.TMax = tmax
        self.Noise = noise
        self.init()
    
    def init(self):
        self.P = self.P0
        changes = np.random.normal(1.0, self.V, size = self.TMax+100)
        changes[0] = 1.0
        
        i = 1
        while i < len(changes) - 6:
            if random.random() < 0.05:
                changes[i] -= self.V*3
                changes[i+1] -= self.V*3
                changes[i+2] -= self.V*3
                changes[i+3] += self.V*3
                changes[i+4] += self.V*3
                i += 5
            else:
                i += 1
        self.Prices = np.cumprod(changes)*self.P
        self.T = self.TWindow+1
        
    def tick(self):
        self.T += 1
        self.P = self.Prices[self.T]
        return self.P

    def deltas(self):
        deltas = self.Prices[self.T-self.TWindow+1:self.T+1]/self.Prices[self.T-self.TWindow:self.T] - 1.0
        return deltas + np.random.normal(size=self.TWindow)*self.Noise
        
class ActionSpace(object):
    def __init__(self, actions):
        self.n = len(actions)
        self.Actions = actions
        
    def sample(self):
        return random.choice(self.Actions)
        
class ObservationSpace(object):
    
    def __init__(self, shape):
        self.shape = shape
        
class Stock3d2uEnv(object):
    
    P = 10.0
    Commission = 0.01
    Noise = 0.0
    
    #DVActions = [0.0, -0.2, -0.02, 0.02, 0.2]
    DVActions = [-0.2, 0.0, 0.2]
    
    def __init__(self, stock, value, ratio, tmax):
        self.Stock = stock
        position = value*ratio/stock.P
        cash = value*(1.0-ratio)
        self.Position0 = position
        self.Cash0 = cash
        self.Position = position        # n shares
        self.Cash = cash
        self.InitialCash = cash
        self.InitialStock = self.stockValue
        self.T = tmax
        self.TMax = tmax
        self.action_space = ActionSpace(self.DVActions)
        self.observation_space = ObservationSpace((self.Stock.TWindow+1,))
        self.Action = None
        self.Reward = None
        self.StockGrowth = 0.0
        #print "Portfolio initialized T=", self.T
        
    def __str__(self):
        return "%d %10.5f -> p:%8.3f C:%8.3f S:%8.3f r:%5.1f%% V=%8.3f G:%5.1f%%,%5.1f%% T:%4d" % (
            self.Action, self.Reward,
            self.Stock.P, self.Cash, self.stockValue, self.ratio*100, self.value,
            self.Stock.growth(self.T1)*100, 
            self.Stock.growth(self.T2)*100,
            self.T
            )
        
    __repr__ = __str__
    
    def info(self):
        return {"T":self.T}
    
    def seed(self, n):
        random.seed(n)
    
    def reset(self):
        self.Stock.init()
        self.Position = self.Position0
        self.Cash = self.Cash0
        self.T = self.TMax
        return self.vectorize()
        
    def step(self, action):
        v0 = self.value
        dv = self.DVActions[action]
        if dv < 0.0:
            self.sell(-dv)
        elif dv > 0.0:
            self.buy(dv)
        p0 = self.Stock.P
        self.Stock.tick()
        self.StockGrowth = self.Stock.P/p0 - 1.0
        r = (self.value - v0)/v0
        self.T -= 1
        #print "step: t=%d, final=%s" % (self.T, self.isFinalState())
        self.Action = action
        self.Reward = r
        return self.vectorize(), r, self.isFinalState(), {}

    def sell(self, dv):
        p = self.Stock.P
        ds = min(dv * self.value, self.stockValue)
        if self.Cash + ds < self.Commission:
            return      # hold
        dp = min(self.Position, ds/p)
        self.Position -= dp
        assert self.Position >= 0.0, "Position is negative: %f" % (self.Position,)
        self.Cash += ds - self.Commission
        assert self.Cash >= 0.0
        
    def buy(self, dv):
        if self.Cash < self.Commission:
            return  # hold
        p = self.Stock.P
        ds = dv * self.value
        ds = min(ds, self.Cash - self.Commission)
        self.Position += ds/p
        self.Cash -= ds + self.Commission
        assert self.Cash >= 0.0
            
    def canSell(self, dv):
        return True
        
    def canBuy(self, dv):
        return True
        
    def validAction(self, action):
        dv = self.DVActions[action]
        if dv < 0.0:    return self.canSell(-dv)
        elif dv > 0.0:  return self.canBuy(dv)
        else:           return True
        
    def validActions(self):
        return [a for a in range(len(self.DVActions)) if self.validAction(a)]
        
    @property
    def value(self):
        return self.Cash + self.stockValue
        
    @property
    def ratio(self):
        return self.stockValue/self.value

    @property
    def stockValue(self):
        return self.Position * self.Stock.P
        
    def isFinalState(self):
        return self.T <= 0
        
        
    def vectorize(self):
        out = np.concatenate(([self.ratio], self.Stock.deltas()))
        return out

    def render(self, mode="human", close=False):
        #raise NotImplementedError
        print self
        pass
        
