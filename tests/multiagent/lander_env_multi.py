from lander_env import LunarLander

class LanderEnvMulti:
    
    def __init__(self):
        self.Env = LunarLander()
        self.Obs = None
        self.Reward = None
        self.Done = False
        self.Info = None
        
    def reset(self, agents):
        assert len(agents) == 1
        self.Done = False
        self.Obs = self.Env.reset()

    ValidActions = [0,1,2,3]
        
    def observe(self, agents):
        assert len(agents) == 1
        return [(agents[0], self.Obs, self.ValidActions, self.Done, self.Info)]
        
    def step(self, actions):
        #print actions
        self.Obs, self.Reward, self.Done, self.Info = self.Env.step(actions[0][1])
        
    def feedback(self, agents):
        return [(agents[0], self.Reward, self.Info)]
        
    def __getattr__(self, name):
        return getattr(self.Env, name)