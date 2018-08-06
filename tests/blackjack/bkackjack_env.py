import random, math, time
import numpy as np
from gym.envs.classic_control import rendering

class Space:
    def __init__(self, shape):
        self.shape = shape
        
class Player:
    
    Values = np.arange(11)+1.0
    
    def __init__(self):
        self.Hand = []      # [assigned_value, ...]
        self.VHand = np.zeros((12,), dtype=np.float32)    # [last_ace,1,2,3...10,11]
        self.Done = False
        self.Reward = 0.0
        self.Double = False
        self.Stay = False
        self.PlayedEpisode = -1
        
    @property
    def lastAce(self):
        return self.VHand[0] != 0

    def score(self):
        return np.sum(self.VHand[1:]*self.Values)
        
    def vhand(self):
        return self.VHand
        
    def addCard(self, value):
        self.Hand.append(value)
        self.VHand[0] = 1 if value == 11 else 0
        self.VHand[value] += 1
        
    def aceTo1(self):
        if self.lastAce:
            self.VHand[0] = 0
            self.VHand[11] -= 1
            self.VHand[1] += 1
            self.Hand[-1] = 1
        
class Dealer(Player):
    
    STAY_LIMIT = 17
    
    def play(self, env):
        if not self.Done and self.score() < self.STAY_LIMIT:
            card = env.dealCard()
            self.addCard(card)
            if card == 11 and self.score() > 21:
                self.aceTo1()
            if self.score() > 21:
                self.Done = True
        
        
class BlackJackEnv(object):
    
    NACTIONS = 6             # stay, ace+stay, hit, ace+hit, double, ace+double
    NPLAYERS = 3            # max 4 players + dealer.
    W = 12
    OBSRVATION_SIZE = W*(NPLAYERS+1)
    observation_space = Space((OBSRVATION_SIZE,))
    action_space = Space((NACTIONS,))
    ALL_ACTIONS = range(NACTIONS)

    def __init__(self):
        self.Deck = []
        for s in range(0,4):
            self.Deck += [2,3,4,5,6,7,8,9,10,10,10,10,11]
        self.Episode = 0
        self.NToPlay = 0
        self.Players = []
        
    def reset(self, players):
        random.shuffle(self.Deck)
        for p in players:
            p.State = Player()
        self.Players = players[:]
        self.Dealer = Dealer()
        self.Over = False
        self.NToPlay = len(players)
        
        for p in players:
            p.State.addCard(self.dealCard())
            
        self.Dealer.addCard(self.dealCard())
            
    def observe(self, players):
        dvector = self.Dealer.vhand()
        pvectors = [(p, p.State.vhand()) for p in players]
        
        vlist = []
        for p, v in pvectors:
            vout = np.zeros((self.NPLAYERS+1, self.W), dtype=np.float32)
            vout[0,:] = v
            vout[1,:] = dvector
            j = 2
            for pp, vv in pvectors:
                if pp is not p:
                    vout[j,:] = vv
                    j += 1
            vlist.append((p, vout.reshape((-1,)), self.ALL_ACTIONS, p.State.Done, {}))
        return vlist
        
    def step(self, agent_actions):
        
        assert len(agent_actions) == 1      # one by one
        
        s = player.State
        assert s.PlayedEpisode < self.Episode

        s.Reward = 0.0
        s.Stay = False
        
        if not s.Done:
        
            if action in (0,1):
                # stay
                s.Stay = True
                if action == 1: s.aceTo1()
                
            elif action in (2,3):
                # hit
                if action == 3: s.aceTo1()
                s.addCard(self.dealCard())
                
            elif action in (4,5):
                s.Double = True
                
            if s.score() > 21:
                s.Done = True
                s.Reward -= -1.0

        s.PlayedEpisode = self.Episode
        self.NToPlay -= 1
        
        if self.NToPlay <= 0:
            self.Dealer.play(self)
            

        return [(player, {})]

    def feedback(self, players):
        all_stay = self.Dealer.Stay
        if all_stay:
            for p in self.Players:
                if not p.State.Done and not p.State.Stay:
                    all_stay = False

        self.Over = all_stay

        out = []
        all_stay = True
        for p in players:
            s = p.State
            out.append((p, s.Reward))
