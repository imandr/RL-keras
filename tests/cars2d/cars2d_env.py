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


class Cars2dEnv(object):
    
    MINXY = 0.0
    MAXXY = 1.0
    WIDTH = MAXXY - MINXY
    
    START_X = 0.2
    START_Y = 0.2
    
    GOAL_X = 0.7
    GOAL_Y = 0.7
    GOAL_XY = np.array([GOAL_X, GOAL_Y])
    
    ACCELERATION = 0.03     # 1/step/step
    EPSILON_XY = 0.03
    EPSILON_V = 0.03
    
    FUEL = 1.0              # 1000 turns max
    FUEL_RATE = 0.01
    
    MAXV = 2.0
    
    NACTIONS = 5            # keep going, accelerate_x, accelerate_y, decelerate_x, decelerate_y
    ALL_ACTIONS = range(NACTIONS)
    
    class State:
        pass
    
    def __init__(self):
        self.Cars = []
        self.observation_space = Space((5,))
        self.actions_space = Space((self.NACTIONS,))
        self.Viewer = rendering.Viewer(self.VIEWPORT, self.VIEWPORT)
        
    def reset(self, cars, random_placement=False):
        for c in cars:
            s = self.State()
            c.State = s
            #s.XY = np.array([self.START_X, self.START_Y]) + (np.random.random((2,))*2-1)*self.EPSILON_XY
            s.XY = np.random.random((2,))*self.WIDTH*0.8 + self.MINXY + self.WIDTH*0.1
            s.V = np.array([0.0, 0.0]) + (np.random.random((2,))*2-1)*self.EPSILON_V
            s.Fuel = self.FUEL
            s.Done = False
            s.Reward = 0.0
        self.Cars = cars[:]
    
    def observe(self, cars):
        return [(c, np.concatenate([c.State.XY, c.State.V, np.array([c.State.Fuel])]), 
                    self.ALL_ACTIONS, c.State.Done, {}) for c in cars]
        
    def step(self, actions):
        out = []
        for c, a in actions:
            s = c.State
            goal_dist_0 = math.sqrt(np.sum(np.square(s.XY - self.GOAL_XY)))
            abs_v_0 = math.sqrt(np.sum(np.square(s.V)))
            reward = 0.0
            if c.State.Fuel <= 0:
                c.State.Done = True
            else:
                a = (a+1) % self.NACTIONS
                if a == 0:
                    s.V *= 0.9
                elif a in (1,2):
                    # accelerate_x or accelerate_y
                    j = a-1
                    s.Fuel -= self.FUEL_RATE
                    if s.V[j] < self.MAXV:
                        s.V[j] += self.ACCELERATION
                elif a in (3,4):
                    # decelerate_x or decelerate_y
                    j = a-3
                    s.Fuel -= self.FUEL_RATE
                    if s.V[j] > -self.MAXV:
                        s.V[j] -= self.ACCELERATION
                s.Fuel -= self.FUEL_RATE
                s.XY += s.V
                if np.any(s.XY < self.MINXY) or np.any(s.XY > self.MAXXY):
                    s.Done = True
                    reward -= 5.0
                else:
                    goal_dist = math.sqrt(np.sum(np.square(s.XY - self.GOAL_XY)))
                    abs_v = math.sqrt(np.sum(np.square(s.V)))
                    reward += goal_dist_0 - goal_dist + abs_v_0 - abs_v
                    if goal_dist < self.EPSILON_XY and abs_v < self.EPSILON_V:
                        reward += 5.0
                        s.Done = True
            s.Reward = reward
            out.append((c, {}))
        return out
                    
    def feedback(self, cars):
        return [(c, c.State.Reward, {}) for c in cars]
            
    VIEWPORT = 600.0
    FPS = 20.0
    SCALE = VIEWPORT/(MAXXY-MINXY)
        
    BodyPoly = [
        (-8, -6),
        (10, 0),
        (-8, 6)
    ]
    BodyColor = (0.1, 0.1, 0.1)
    
    IndicatorPoly = [
        (-2, -2),
        (-2, 2),
        (2, 2),
        (2, -2)
    ]
    
    SuccessColor = (0.0, 1.0, 0.0)
    FailColor = (1.0, 0.0, 0.0)

    def shift_rotate(self, poly, angle, move):
        rotation = np.array([
            (math.cos(angle), math.sin(angle)),
            (-math.sin(angle), math.cos(angle))
        ])
        return np.array(poly).dot(rotation)+np.array(move*self.SCALE)
    
    def render(self, mode = "human"):
        t0 = time.time()
        t1 = t0 + 1.0/self.FPS
        for c in self.Cars:
            s = c.State
            phi = math.atan2(s.V[1], s.V[0])
            color = self.BodyColor if not s.Done else (
                self.SuccessColor if s.Reward > 0.0 else self.FailColor
            )
            self.Viewer.draw_polygon(self.shift_rotate(self.BodyPoly, phi, s.XY), color=color)
            if abs(s.Reward) > 0.005:
                self.Viewer.draw_polygon(self.shift_rotate(self.IndicatorPoly, phi, s.XY), 
                    color = (1,0,0) if s.Reward < 0 else (0,1,0))
        self.Viewer.draw_circle(
                radius=self.EPSILON_XY*self.SCALE, 
                color=(0.0, 0.0, 1.0),
                filled=False
                ).add_attr(
            rendering.Transform(translation=np.array([self.GOAL_X, self.GOAL_Y])*self.SCALE))

        self.Viewer.draw_circle(
                radius=self.EPSILON_XY*self.SCALE, 
                color=(0.0, 0.0, 1.0),
                filled=False
                ).add_attr(
            rendering.Transform(translation=np.array([self.GOAL_X, self.GOAL_Y])*self.SCALE))


        self.Viewer.render()
        time.sleep(max(0, t1 - time.time()))

if __name__ == '__main__':        
    
    class Car:  pass
    
    env = Cars2dEnv()
    nactions = env.actions_space.shape[-1]
    
    while True:
        cars = [Car() for _ in xrange(2)]
        env.reset(cars)
        all_done = False
        while not all_done:
            observations = env.observe(cars)
            actions = [(c,random.randint(0, nactions-1)) for c, o, v, d, i in observations if not d]
            all_done = len(actions) == 0
            if not all_done:
                env.step(actions)
                feedback = env.feedback([c for c, a in actions])
            env.render()
        time.sleep(1)
                
class CarsRadEnv(object):
    
    MINXY = 0.0
    MAXXY = 1.0
    WIDTH = MAXXY - MINXY
    
    START_X = 0.2
    START_Y = 0.2
    
    GOAL_X = 0.7
    GOAL_Y = 0.7
    GOAL_XY = np.array([GOAL_X, GOAL_Y])
    
    ACCELERATION = 0.03     # 1/step/step
    EPSILON_XY = 0.03
    EPSILON_V = 0.03
    
    DPHI = 10.0/180.0*math.pi
    
    FUEL = 1.0              # 1000 turns max
    FUEL_RATE = 0.007
    
    MAXV = 2.0
    MINV = 0.0
    
    NACTIONS = 5            # keep idle, accelerate, decelerate, turn right, turn left
    ALL_ACTIONS = range(NACTIONS)
    
    class State:
        pass
    
    def __init__(self):
        self.Cars = []
        self.observation_space = Space((7,))
        self.actions_space = Space((self.NACTIONS,))
        self.Viewer = rendering.Viewer(self.VIEWPORT, self.VIEWPORT)
        
    def reset(self, cars, random_placement=False):
        for c in cars:
            if not hasattr(c, "State"):
                c.State = self.State()
            s = c.State
            #s.XY = np.array([self.START_X, self.START_Y]) + (np.random.random((2,))*2-1)*self.EPSILON_XY
            s.XY = np.random.random((2,))*self.WIDTH*0.8 + self.MINXY + self.WIDTH*0.1
            s.phi = (random.random()*2-1)
            s.V = random.random()*self.EPSILON_V
            s.Fuel = self.FUEL
            s.Done = False
            s.Reward = 0.0
            if not hasattr(s, "Color"):   
                color = np.random.random((3,))
                cmin = min(color)
                cmax = max(color)
                color = 0.1 + (color-cmin)/(cmax-cmin)*0.5
                s.Color = color
        self.Cars = cars[:]
    
    def observe(self, cars):
        out = []
        for car in cars:
            s = car.State
            dxy = self.GOAL_XY - s.XY
            dist = math.sqrt(np.sum(dxy*dxy))
            alpha = angle_in_range(math.atan2(dxy[1], dxy[0])-s.phi)
            out.append((car, np.array([
                    s.XY[0], s.XY[1], s.V, s.Fuel, s.phi/math.pi, dist, alpha/math.pi
                ]), self.ALL_ACTIONS, s.Done, {}
            ))
        return out
        
    def step(self, actions):
        out = []
        for c, a in actions:
            s = c.State
            goal_dist_0 = math.sqrt(np.sum(np.square(s.XY - self.GOAL_XY)))
            abs_v_0 = s.V
            reward = 0.0
            if s.Fuel <= 0:
                s.Done = True
                reward -= 1.0
            else:
                if a == 0:
                    s.V *= 0.9
                elif a == 1:
                    s.Fuel -= self.FUEL_RATE
                    s.V = min(self.MAXV, s.V+self.ACCELERATION)
                elif a == 2:
                    s.V = max(self.MINV, s.V-self.ACCELERATION)
                    s.Fuel -= self.FUEL_RATE
                elif a in (3,4):
                    # decelerate_x or decelerate_y
                    j = a-3
                    dphi = [-self.DPHI, self.DPHI][j]
                    s.phi = angle_in_range(s.phi+dphi)
                    s.V *= 0.5
                s.Fuel -= self.FUEL_RATE
                s.XY[0] += s.V * math.cos(s.phi)
                s.XY[1] += s.V * math.sin(s.phi)
                if np.any(s.XY < self.MINXY) or np.any(s.XY > self.MAXXY):
                    s.Done = True
                    reward -= 5.0
                else:
                    goal_dist = math.sqrt(np.sum(np.square(s.XY - self.GOAL_XY)))
                    abs_v = math.sqrt(np.sum(np.square(s.V)))
                    reward += goal_dist_0 - goal_dist #+ abs_v_0 - abs_v
                    if goal_dist < self.EPSILON_XY and abs_v < self.EPSILON_V:
                        reward += 5.0
                        s.Done = True
            s.Reward = reward
            out.append((c, {}))
        return out
                    
    def feedback(self, cars):
        return [(c, c.State.Reward, {}) for c in cars]
            
    VIEWPORT = 600.0
    FPS = 20.0
    SCALE = VIEWPORT/(MAXXY-MINXY)
        
    BodyPoly = [
        (-8, -6),
        (10, 0),
        (-8, 6)
    ]
    BodyColor = (0.1, 0.1, 0.1)
    
    IndicatorPoly = [
        (-2, -2),
        (-2, 2),
        (2, 2),
        (2, -2)
    ]
    
    SuccessColor = (0.0, 1.0, 0.0)
    FailColor = (1.0, 0.0, 0.0)

    def shift_rotate(self, poly, angle, move):
        rotation = np.array([
            (math.cos(angle), math.sin(angle)),
            (-math.sin(angle), math.cos(angle))
        ])
        return np.array(poly).dot(rotation)+np.array(move*self.SCALE)
    
    def render(self, mode = "human"):
        t0 = time.time()
        t1 = t0 + 1.0/self.FPS
        for c in self.Cars:
            s = c.State
            phi = s.phi
            color = self.BodyColor if not s.Done else (
                self.SuccessColor if s.Reward > 0.0 else self.FailColor
            )
            self.Viewer.draw_polygon(self.shift_rotate(self.BodyPoly, phi, s.XY), color=s.Color, filled=not s.Done)
            if abs(s.Reward) > 0.005 and not s.Done:
                self.Viewer.draw_polygon(self.shift_rotate(self.IndicatorPoly, phi, s.XY), 
                    color = (1,0,0) if s.Reward < 0 else (0,1,0))
        self.Viewer.draw_circle(
                radius=self.EPSILON_XY*self.SCALE, 
                color=(0.0, 0.0, 1.0),
                filled=False
                ).add_attr(
            rendering.Transform(translation=np.array([self.GOAL_X, self.GOAL_Y])*self.SCALE))

        self.Viewer.draw_circle(
                radius=self.EPSILON_XY*self.SCALE, 
                color=(0.0, 0.0, 1.0),
                filled=False
                ).add_attr(
            rendering.Transform(translation=np.array([self.GOAL_X, self.GOAL_Y])*self.SCALE))


        self.Viewer.render()
        time.sleep(max(0, t1 - time.time()))

if __name__ == '__main__':        
    
    class Car:  pass
    
    env = Cars2dEnv()
    nactions = env.actions_space.shape[-1]
    
    while True:
        cars = [Car() for _ in xrange(2)]
        env.reset(cars)
        all_done = False
        while not all_done:
            observations = env.observe(cars)
            actions = [(c,random.randint(0, nactions-1)) for c, o, v, d, i in observations if not d]
            all_done = len(actions) == 0
            if not all_done:
                env.step(actions)
                feedback = env.feedback([c for c, a in actions])
            env.render()
        time.sleep(1)
                
        
        
        
        