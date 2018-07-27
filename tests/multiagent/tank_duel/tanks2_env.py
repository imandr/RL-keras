import random, math, time
import numpy as np
from gym.envs.classic_control import rendering

class Space:
    def __init__(self, shape):
        self.shape = shape
        
def angle_in_range(a):
    while a >= math.pi:
        a -= math.pi*2
    while a < -math.pi:
        a += math.pi*2
    return a

class TankDuelEnv(object):
    
    SIZE = 100.0
    SPEED = 2.0
    RANGE = 30.0
    HIT_SIGMA = 2.0
    TURN = 1.0*math.pi/180      
    TURRET_TURN = 1.0*math.pi/180   
    
    NACTIONS = 8   # noop, fire, 
                    # forward slow, forward fast, 
                    # turn left fast, turn left slow, turn right slow, turn right fast, 
                    
    ALL_ACTIONS = range(NACTIONS)
    FIRE_ACTION = 1
    
    VIEWPORT = 600
    FPS = 100.0
    TMAX = 500.0
    
    def __init__(self):
        self.Tanks = []
        self.Actions = {}
        self.Viewer = rendering.Viewer(self.VIEWPORT, self.VIEWPORT)
        self.observation_space = Space((10,))
        self.action_space = Space((self.NACTIONS,))
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

        nalive = sum([1 for t in self.Tanks if not t.killed], 0)
        game_over = nalive <= 1 or self.T <= 0

        for i, t in enumerate(self.Tanks):
            t1 = self.Tanks[1-i]
            dxy = t1.xy - t.xy
            
            alpha = math.atan2(dxy[1], dxy[0])
            dist = math.sqrt(np.sum(dxy**2))
            
            observation = np.array([
                # self position relative to edges
                t.xy[0]/self.RANGE,
                t.xy[1]/self.RANGE,
                (self.SIZE-t.xy[0])/self.RANGE,
                (self.SIZE-t.xy[1])/self.RANGE,
                # self rangles
                t.phi/math.pi,
                t.theta/math.pi,
                # the other tank position
                dist/self.RANGE,
                angle_in_range(alpha-t.phi-t.theta)/math.pi,
                angle_in_range(alpha+math.pi-t1.phi-t1.theta)/math.pi,
                self.T/self.TMAX
            ])
            
            if i == 0:
                assert observation.shape == self.observation_space.shape
            out.append((t, observation, self.ALL_ACTIONS, game_over, {}))
        return out
            
    def step(self, agents_actions, agent=None):
        if not isinstance(agents_actions, list):
            agents_actions = [(agent, agents_actions)]
            
        for t, _ in agents_actions:
            t.reward = 0.0

        # all fire actions first
        for t, a in agents_actions:
            self.Actions[id(t)] = a    # for rendering
            if a == self.FIRE_ACTION:    # fire
                # find the other tank
                other = None
                for t1 in self.Tanks:
                    if not t1 is t:
                        other = t1
                        break
                dxy = other.xy - t.xy
                alpha = math.atan2(dxy[1], dxy[0])
                delta = angle_in_range(alpha - t.phi - t.theta)
                dist = math.sqrt(np.sum(dxy*dxy))
                if dist <= self.RANGE and abs(delta) < math.pi/2 and abs(dist*delta) <= self.HIT_SIGMA:
                    print "===> hit: alpha=", alpha, "  phi+theta:", t.phi + t.theta, "  delta:", delta, "  dist:", dist
                    other.killed = True
                    other.reward -= 5.0
                    t.reward += 5.0
                else:
                    t.reward += -0.01
        
        
        for t, a in agents_actions:
            
            if t.killed or a == 0 or a == self.FIRE_ACTION:      # noop
                pass
                    
            elif a in (2,3):
                # move
                #dcenter = t.xy - self.SIZE/2
                #dist_from_center0 = math.sqrt(np.sum(dcenter**2))
                dist = (0.2, 1.2)[a-2]
                dx = dist*math.cos(t.phi)
                dy = dist*math.sin(t.phi)
                t.xy[0] += dx
                t.xy[1] += dy
                if t.xy[0] < 0.0 or t.xy[0] > self.SIZE or \
                        t.xy[1] < 0.0 or t.xy[1] > self.SIZE:
                    t.reward -= 5.0
                    t.killed = True
                
                
            elif a in (4,5,6,7):
                # turn
                t.phi = angle_in_range(t.phi + (-5,-1,1,5)[a-4]*self.TURN)
                
        return [(agent, {}) for agent, action in agents_actions]
                
    def feedback(self, agents):
        if not  isinstance(agents, list):
            agents = [agents]
        out = []
        self.T -= 1
        return [(t, t.reward, {}) for t in agents]
        
    
    SCALE = VIEWPORT/SIZE
        
    BodyPoly = [
        (-12, -6),
        (8, -6),
        (8, 6),
        (-12, 6)
    ]
    
    BodyColor = (0.2, 0.8, 0.4)
    
    TowerPoly = [
        (-5,-5),
        (4,-3),
        (4,3),
        (-5,5)
    ]
    
    TowerColor = (0.1, 0.4, 0.2)
    
    CannonPoly = [
        (3,-1),
        (15,-1),
        (15,1),
        (3,1)
    ]
    
    DeadColor = (0.5, 0.1, 0.1)
    
    CannonColor = (0.1, 0.4, 0.2)
    
    FirePoly = [
        (17, -0.5),
        (RANGE*SCALE, -0.5),
        (RANGE*SCALE, 0.5),
        (17, 0.5)
    ]
    
    FireColor = (1.0, 0.2, 0.)
    
    def shift_rotate(self, poly, angle, move):
        rotation = np.array([
            (math.cos(angle), math.sin(angle)),
            (-math.sin(angle), math.cos(angle))
        ])
        return np.array(poly).dot(rotation)+np.array(move*self.SCALE)
    
        
    def render(self, mode="human"):
        t0 = time.time()
        t1 = t0 + 1.0/self.FPS
        killed = False
        for t in self.Tanks:
            self.Viewer.draw_polygon(self.shift_rotate(self.BodyPoly, t.phi, t.xy), 
                color=self.DeadColor if t.killed else self.BodyColor)
            self.Viewer.draw_polygon(self.shift_rotate(self.TowerPoly, t.phi+t.theta, t.xy), 
                    color=self.DeadColor if t.killed else self.TowerColor)
            self.Viewer.draw_polygon(self.shift_rotate(self.CannonPoly, t.phi+t.theta, t.xy),
                    color=self.DeadColor if t.killed else self.CannonColor)
            if self.Actions.get(id(t)) == self.FIRE_ACTION:
                self.Viewer.draw_polygon(self.shift_rotate(self.FirePoly, t.phi+t.theta, t.xy),
                    color = self.FireColor)
            killed = killed or t.killed
        self.Viewer.render()
        if killed:
            time.sleep(1.0)
        else:
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
            
        