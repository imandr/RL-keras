import random, math, time
import numpy as np
from gym.envs.classic_control import rendering
#import pyglet

class Space:
    def __init__(self, shape):
        self.shape = shape
        
def angle_in_range(a):
    while a >= math.pi:
        a -= math.pi*2
    while a < -math.pi:
        a += math.pi*2
    return a

class TankTargetEnv(object):
    
    SIZE = 100.0
    SPEED = 2.0
    RANGE = 30.0
    TARGET_RADIUS = 3.0
    TURN = 1.0*math.pi/180      
    TURRET_TURN = 1.0*math.pi/180   
    
    NACTIONS = 7   # fire, 
                    # forward slow, forward fast, 
                    # turn left fast, turn left slow, turn right slow, turn right fast, 
                    # turret left slow, turret left fast,
                    
    ALL_ACTIONS = range(NACTIONS)
    FIRE_ACTION = 0
    
    VIEWPORT = 600
    FPS = 200.0
    TMAX = 500
    
    class TankState(object):
        
        def move(self, dist):
            dx = dist*math.cos(self.phi)
            dy = dist*math.sin(self.phi)
            self.xy[0] += dx
            self.xy[1] += dy
            if self.xy[0] < 0.0 or self.xy[0] > TankTargetEnv.SIZE or \
                    self.xy[1] < 0.0 or self.xy[1] > TankTargetEnv.SIZE:
                self.reward -= 1.0
                self.Done = True
                
        def targetDist(self, target):
            dxy = target - self.xy
            return math.sqrt(dxy[0]**2+dxy[1]**2)
            
        def targetAngle(self, target):
            dxy = target - self.xy
            return math.atan2(dxy[1], dxy[0])
            
        def observation(self, t, target):
            #print "observation: t=", t
            return np.array([
                float(t)/TankTargetEnv.TMAX,        # time remaining
                self.xy[0]/TankTargetEnv.RANGE,     # distance to walls
                self.xy[1]/TankTargetEnv.RANGE,
                (TankTargetEnv.SIZE-self.xy[0])/TankTargetEnv.RANGE,
                (TankTargetEnv.SIZE-self.xy[1])/TankTargetEnv.RANGE,
                self.phi/math.pi,                   # angle
                self.targetDist(target)/TankTargetEnv.RANGE,    # distance to target
                angle_in_range(self.targetAngle(target) - self.phi - self.theta)/math.pi   # relative angle to target
            ])
            
        def turn(self, delta):
            self.phi = angle_in_range(self.phi + delta)
        
        def turret_turn(self, delta):
            self.theta = angle_in_range(self.theta + delta)
            
    
    def __init__(self):
        self.Tanks = []
        self.Actions = {}
        self.Viewer = rendering.Viewer(self.VIEWPORT, self.VIEWPORT)
        #w=self.Viewer.window
        #cfg = pyglet.gl.Config(accum_red_size=8, accum_green_size=8, accum_blue_size=8)
        
        self.observation_space = Space((8,))
        self.action_space = Space((self.NACTIONS,))
        self.Target = None
        self.T = self.TMAX
        self.Over = False
        
    def over(self):
        return self.T <= 0

    def reset(self, tanks):
        #assert len(tanks) == 2
        self.Tanks = []
        self.Actions = {}
        self.T = self.TMAX
        self.Over = False
        for t in tanks:
            s = self.TankState()
            t.State = s
            s.xy = np.random.uniform(high=self.SIZE, size=(2,))
            s.phi = np.random.uniform(low=-math.pi, high=math.pi)
            self.Target = np.random.uniform(low=self.SIZE*0.1, high=self.SIZE*0.9, size=(2,))
            s.theta = 0.0   # turret relative to body
            s.reward = 0.0
            s.last_fire = False
            s.Done = False
            s.Hit = False
            s.Shots = 0
            self.Tanks.append(t)
    
    def observe(self, tanks):
        out = []

        self.Over = self.Over or self.T <= 0

        for i, t in enumerate(self.Tanks):
            s = t.State            
            observation = s.observation(self.T, self.Target)
            if i == 0:
                assert observation.shape == self.observation_space.shape
            out.append((t, observation, self.ALL_ACTIONS, s.Done or self.Over, {}))
        return out
            
    def step(self, agents_actions, agent=None):
        
        if not isinstance(agents_actions, list):
            agents_actions = [(agent, agents_actions)]
            
        for t in self.Tanks:
            t.State.reward = 0.0

        if not self.Over:

            for t, a in agents_actions:
                s = t.State
                s.Hit = False
                self.Actions[id(t)] = a    # for rendering

                if a == 0:    # fire
                    # find the other tank
                    s.Shots += 1
                    dist = s.targetDist(self.Target)
                    if dist <= self.RANGE:
                        alpha = s.targetAngle(self.Target)
                        delta = angle_in_range(alpha - s.phi - s.theta)
                        if abs(delta) < math.pi/2 and \
                                abs(math.sin(delta)*dist) < self.TARGET_RADIUS:
                            print "=--> hit: alpha=", alpha, "  phi+theta:", s.phi + s.theta, "  delta:", delta, "  dist:", dist
                            s.Hit = True        # for rendering
                    if s.Hit:
                        s.Done = True
                        s.reward += 5.0
                    else:
                        s.reward -= 0.01
                    
                elif a in (1,2):
                    # move
                    dist = (0.2, 1.2)[a-1]
                    s.move(dist)
                
                elif a in (3,4,5,6):
                    # turn
                    s.turn((-7,-2,2,7)[a-3]*self.TURN)
                #print a, t.reward
                
        return [(agent, {}) for agent, action in agents_actions]
                
    def feedback(self, agents):
        if not  isinstance(agents, list):
            agents = [agents]
        self.T -= 1
        return [(t, t.State.reward, {}) for t in agents]
        
    
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
        hit = False
        for t in self.Tanks:
            if t.State.Hit:
                hit = True
                t.State.Hit = False
                break
                
        # draw the target
        if hit:
            self.Viewer.draw_circle(radius=self.TARGET_RADIUS*self.SCALE,
                color=(1.0, 0.0, 0.0)
                ).add_attr(rendering.Transform(translation=self.Target*self.SCALE))
            self.Viewer.draw_circle(radius=self.TARGET_RADIUS*self.SCALE*4/5,
                color=(1.0, 0.5, 0.0)
                ).add_attr(rendering.Transform(translation=self.Target*self.SCALE))
            self.Viewer.draw_circle(radius=self.TARGET_RADIUS*self.SCALE*2/5,
                color=(1.0, 0.8, 0.0)
                ).add_attr(rendering.Transform(translation=self.Target*self.SCALE))
        else:
            self.Viewer.draw_circle(radius=self.TARGET_RADIUS*self.SCALE,
                color=(0.0, 0.0, 1.0)
                ).add_attr(rendering.Transform(translation=self.Target*self.SCALE))
            self.Viewer.draw_circle(radius=self.TARGET_RADIUS*self.SCALE*4/5,
                color=(1.0, 1.0, 1.0)
                ).add_attr(rendering.Transform(translation=self.Target*self.SCALE))
            self.Viewer.draw_circle(radius=self.TARGET_RADIUS*self.SCALE*2/5,
                color=(0.0, 0.0, 1.0)
                ).add_attr(rendering.Transform(translation=self.Target*self.SCALE))
                
        # the tanks

        for t in self.Tanks:
            s = t.State
            self.Viewer.draw_polygon(self.shift_rotate(self.BodyPoly, s.phi, s.xy), 
                color=self.BodyColor, filled=not s.Done)
            self.Viewer.draw_polygon(self.shift_rotate(self.TowerPoly, s.phi+s.theta, s.xy), 
                    color=self.TowerColor, filled=not s.Done)
            self.Viewer.draw_polygon(self.shift_rotate(self.CannonPoly, s.phi+s.theta, s.xy),
                    color=self.CannonColor, filled=not s.Done)
            if self.Actions.get(id(t)) == self.FIRE_ACTION:
                self.Viewer.draw_polygon(self.shift_rotate(self.FirePoly, s.phi+s.theta, s.xy),
                    color = self.FireColor, filled=not s.Done)
            
                
        self.Viewer.render()
        if hit:
            time.sleep(0.5)
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
            
        