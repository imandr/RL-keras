import gym, random
import numpy as np



class SecretarySelectionEnv(gym.env):

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
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.YTrue = np.random.random((self.NCandidates,))
        while self.YTrue[0] == 0.0:
            self.YTrue[0] = random.random()
        self.YTrue = np.asarray(self.YTrue/self.YTrue[0], dtype=np.float32)
        self.Y = np.asarray(np.ones((self.NCandidates,)), dtype=np.float32) * -1.0
        self.I = 0
        self.Hired = None
        return self.step(0)[0]
        
    def step(self, action):
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
        
        reward = 0.0
        if hired_y is not None:
            ymax = max(self.YTrue)
            reward = 10.0 if hired_y == ymax else hired_y/ymax
        obs = np.concatenate((
            [np.float32(self.I)/np.float32(self.NCandidates)],
            self.Y
            ))
        return obs, reward, hired_y is not None, {
            "y": self.YTrue,
            "n_seen": self.I
        }

    def render(self):
        hired_y, hired_i = None, None
        if self.Hired:  hired_y, hired_i = self.Hired
        items = ["%7.3f %1s" % (y, '<' if self.I == i+1 else ' ') for i, y in enumerate(self.YTrue)]
        print "%s %s" % (
            " ".join(items),
            "hired %.3f" % (self.hired_y/max(self.YTrue),) if hired_y is not None else ""
        )
            
    