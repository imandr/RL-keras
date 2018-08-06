import random, math
import numpy as np

class Stock:
    def __init__(self, price, volatility, noise, tmax):
        self.P0 = price
        self.V = volatility/math.sqrt(float(tmax))
        self.TMax = tmax
        self.Noise = noise
        self.init()
    
    def init(self, ):
        self.P = self.P0
        changes = np.random.normal(1.0, self.V, size = self.TMax+100)
        changes[0] = 1.0
        self.Future = np.cumprod(changes)*self.P
        self.T = 0
        
    def tick(self):
        self.T += 1
        self.P = self.Future[self.T]
        return self.P

    def growth(self, t):
        return self.Future[self.T+t]/self.P - 1.0 + random.gauss(0.0, self.Noise)
        
class ActionSpace(object):
    def __init__(self, actions):
        self.n = len(actions)
        self.Actions = actions
        
    def sample(self):
        return random.choice(self.Actions)
        
class ObservationSpace(object):
    
    def __init__(self, shape):
        self.shape = shape
        
class PredictedEnv(object):
    
    P = 10.0
    Commission = 0.01
    Noise = 0.0
    
    T1 = 1
    T2 = 5
    
    #DVActions = [0.0, -0.2, -0.02, 0.02, 0.2]
    DVActions = [-0.2, -0.02, 0.0, 0.02, 0.2]
    
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
        self.observation_space = ObservationSpace((3,))
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
        assert self.canSell(dv)
        p = self.Stock.P
        ds = dv * self.value
        self.Position -= ds/p
        self.Cash += ds - self.Commission
        
    def buy(self, dv):
        assert self.canBuy(dv)
        p = self.Stock.P
        ds = dv * self.value
        self.Position += ds/p
        self.Cash -= ds + self.Commission
            
    def canSell(self, dv):
        ds = self.value * dv
        return self.stockValue >= ds and self.Cash + ds >= self.Commission
        
    def canBuy(self, dv):
        ds = self.value * dv
        return self.Cash >= ds + self.Commission
        
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
        inv = self.stockValue
        out = np.array(
            (   
                self.ratio,
                self.Stock.growth(self.T1), 
                self.Stock.growth(self.T2)
            )
        )
        #print "vectorize: %s -> %s" % (self, out)
        return out

    def render(self, mode="human", close=False):
        #raise NotImplementedError
        print self
        pass
        
