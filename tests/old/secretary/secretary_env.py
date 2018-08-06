import gym, random
import numpy as np
from gym import spaces



class SecretarySelectionEnv(gym.Env):

    YMAX = 10.0
    
    # Actions:
    #   0: pass
    #   1: hire
    
    def __init__(self, ncandidates):
        self.NCandidates = ncandidates
        self.YTrue = None        
        self.Y = None
        self.I = 0      # n seen
        self.Hired = None
        
        high = np.array([np.inf]*(ncandidates+1))  # useful range is -1 .. +1, but spikes can be higher
        low = np.zeros((ncandidates+1,))
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Discrete(2)        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        done = False
        while not done:
            self.YTrue = np.random.random((self.NCandidates,))
            done = self.YTrue[0] > 0.1
        self.YTrue = np.asarray(self.YTrue/self.YTrue[0], dtype=np.float32)
        self.Y = np.asarray(np.ones((self.NCandidates,)), dtype=np.float32) * -1.0
        self.I = 0
        self.Hired = None
        return self.step(0)[0]
        
    def step(self, action):
        #print "action:", action
        hired_y = None
        reward = 0.0
        if self.I >= self.NCandidates:
            action = 1      # must hire the last one
        if action == 0:
            # pass
            self.Y[self.I] = self.YTrue[self.I]
            self.I += 1
        else:
            # hire
            hired_y = self.YTrue[self.I-1]
            self.Hired = (self.I-1, hired_y)
            
        done = hired_y is not None
        
        reward = 0.0
        if done:
            ymax = max(self.YTrue)
            reward = 10.0 if hired_y == ymax else hired_y/ymax
        obs = np.concatenate((
            [np.float32(self.I)/np.float32(self.NCandidates)],
            self.Y
            ))
        #print obs
        return obs, reward, done, {
            "y": self.YTrue,
            "n_seen": self.I,
            "ratio": hired_y/ymax if done else None
        }

    def render(self, mode="human"):
        hired_i, hired_y = None, None
        if self.Hired:  hired_i, hired_y = self.Hired
        items = ["%6.3f%1s" % (y, '<' if self.I == i+1 else ' ') for i, y in enumerate(self.YTrue)]
        print "%s %s" % (
            " ".join(items),
            "hired %.3f (hired=%f, max=%f)" % (hired_y/max(self.YTrue), hired_y, max(self.YTrue)) if hired_y is not None else ""
        )
            
    