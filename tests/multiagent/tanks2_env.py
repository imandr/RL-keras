import random, math, time
import numpy as np
from gym.envs.classic_control import rendering

class Space:
    def __init__(self, shape):
        self.shape = shape

class TankDuelEnv(object):
    
    SIZE = 100.0
    SPEED = 2.0
    RANGE = 50.0
    HIT_SIGMA = 3.0
    TURN = 1.0*math.pi/180      
    TURRET_TURN = 1.0*math.pi/180   
    
    NACTIONS = 13   # noop, fire, 
                    # forward slow, forward fast, reverse, 
                    # turn left fast, turn left slow, turn right slow, turn right fast, 
                    # turret left slow, turret left fast,
                    # turret right slow, turret right fast, 
                    
    ALL_ACTIONS = range(NACTIONS)
    FIRE_ACTION = 1
    
    VIEWPORT = 600
    FPS = 100.0
    TMAX = 1000
    
    def __init__(self):
        self.Tanks = []
        self.Actions = {}
        self.Viewer = rendering.Viewer(self.VIEWPORT, self.VIEWPORT)
        self.observation_space = Space((10,))
        self.actions_space = Space((self.NACTIONS,))
        self.T = self.TMAX

    def reset(self, tanks):
        assert len(tanks) == 2
        self.Tanks = []
        self.Actions = {}
        self.T = self.TMAX
        for t in tanks:
            t.xy = np.random.uniform(high=self.SIZE, size=(2,))
            t.phi = np.random.uniform(low=-math.pi, high=math.pi)
            t.theta = 0.0   # turret relative to body
            t.killed = False
            t.reward = 0.0
            self.Tanks.append(t)
    
    def observe(self, tanks):
        out = []
        for i, t in enumerate(self.Tanks):
            t1 = self.Tanks[1-i]
            dxy = t1.xy - t.xy
            
            observation = np.array([
                # self position relative to edges
                1.0/t.xy[0],
                1.0/t.xy[1],
                1.0/(self.SIZE-t.xy[0]),
                1.0/(self.SIZE-t.xy[1]),
                # self rangles
                t.phi/math.pi,
                t.theta/math.pi,
                # the other tank position
                dxy[0]/self.RANGE,
                dxy[1]/self.RANGE,
                # the other tank angles
                t1.phi/math.pi,
                t1.theta/math.pi
            ])
            
            if i == 0:
                assert observation.shape == self.observation_space.shape
            out.append((t, observation, self.ALL_ACTIONS))
        return out
            
    def step(self, agents_actions, agent=None):
        if not isinstance(agents_actions, list):
            agents_actions = [(agent, agents_actions)]
        
        for t, a in agents_actions:
            
            if t.killed:    continue
            
            self.Actions[id(t)] = a    # for rendering
            
            
            if a == 0:      # noop
                pass
            elif a == self.FIRE_ACTION:    # fire
                # find the other tank
                other = None
                for t1 in self.Tanks:
                    if not t1 is t:
                        other = t1
                        break
                dxy = other.xy - t.xy
                alpha = math.atan2(dxy[1], dxy[0])
                delta = alpha - t.phi - t.theta
                while delta > math.pi: delta -= math.pi*2
                while delta < -math.pi: delta += math.pi*2
                dist = math.sqrt(np.sum(dxy*dxy))
                if dist <= self.RANGE and abs(delta) < math.pi and abs(dist*delta) <= self.HIT_SIGMA:
                    print "hit: alpha=", alpha, "  phi+theta:", t.phi + t.theta, "  delta:", delta, "  dist:", dist
                    other.killed = True
                    other.reward = -1.0
                    t.reward = 1.0
                else:
                    t.reward = -0.01
                    
            elif a in (2,3,4):
                dcenter = t.xy - self.SIZE/2
                dist_from_center0 = math.sqrt(np.sum(dcenter**2))
                dist = (0.2, 1.2, -0.2)[a-2]
                dx = dist*math.cos(t.phi)
                dy = dist*math.sin(t.phi)
                if t.xy[0] + dx < self.SIZE and t.xy[0] + dx > 0.0 and \
                            t.xy[1] + dy < self.SIZE and t.xy[1] + dy > 0.0:
                    t.xy[0] += dx
                    t.xy[1] += dy
                    dcenter = t.xy - self.SIZE/2
                    dist_from_center1 = math.sqrt(np.sum(dcenter**2))
                    t.reward = (dist_from_center0-dist_from_center1)/10.0
                    #print "reward:", t.reward
                
            elif a in (5,6,7,8):
                # turn
                t.phi += (-5,-1,1,5)[a-5]*self.TURN
            elif a in (9,10,11,12):
                # turret turn
                t.theta += (-10,-2,2,10)[a-9]*self.TURRET_TURN
                
    def feedback(self, agents):
        if not  isinstance(agents, list):
            agents = [agents]
        out = []
        nalive = sum([1 for t in self.Tanks if not t.killed], 0)
        self.T -= 1
        game_over = nalive <= 1 or self.T <= 0
        for t in agents:
            reward = t.reward
            t.reward = 0.0
            done = game_over or t.killed
            out.append((t, reward, done, {}))
        return out
        
    
    SCALE = VIEWPORT/SIZE
        
    BodyPoly = [
        (-10, -6),
        (10, -6),
        (10, 6),
        (-10, 6)
    ]
    
    BodyColor = (0.0, 0.4, 0.1)
    
    TowerPoly = [
        (-5,-4),
        (4,-4),
        (4,4),
        (-4,4)
    ]
    
    TowerColor = (0.0, 0.1, 0.4)
    
    CannonPoly = [
        (3,-1),
        (15,-1),
        (15,1),
        (3,1)
    ]
    
    CannonColor = (0.0, 0.0, 0.0)
    
    FirePoly = [
        (17, -1),
        (17+RANGE*SCALE, -1),
        (17+RANGE*SCALE, 1),
        (17, 1)
    ]
    
    FireColor = (1.0, 0.8, 0.4)
    
    def shift_rotate(self, poly, angle, move):
        rotation = np.array([
            (math.cos(angle), math.sin(angle)),
            (-math.sin(angle), math.cos(angle))
        ])
        return np.array(poly).dot(rotation)+np.array(move*self.SCALE)
    
        
    def render(self, mode="human"):
        t0 = time.time()
        t1 = t0 + 1.0/self.FPS
        
        for t in self.Tanks:
            self.Viewer.draw_polygon(self.shift_rotate(self.BodyPoly, t.phi, t.xy), color=self.BodyColor)
            self.Viewer.draw_polygon(self.shift_rotate(self.TowerPoly, t.phi+t.theta, t.xy), 
                    color=self.TowerColor)
            self.Viewer.draw_polygon(self.shift_rotate(self.CannonPoly, t.phi+t.theta, t.xy),
                    color=self.CannonColor)
            if self.Actions[id(t)] == self.FIRE_ACTION:
                self.Viewer.draw_polygon(self.shift_rotate(self.FirePoly, t.phi+t.theta, t.xy),
                    color = self.FireColor)
        self.Viewer.render()
        time.sleep(max(0, t1 - time.time()))
        
            
if __name__=='__main__':
    import time
    
    class Tank:
        pass
        
    env = TankDuelEnv()
    
    for episide in range(100):
        tanks = [Tank(), Tank()]
        env.reset(tanks)
        nactive = 2
        while nactive:
            env.observe(tanks)
            actions = np.random.randint(0, high=env.NACTIONS, size=(2,))
            env.step(zip(tanks, actions))
            #print env.Actions
            nactive = len(tanks)
            env.render()
            for t, reward, done, info in env.feedback(tanks):
                if done: nactive -= 1
        time.sleep(1)
            
        