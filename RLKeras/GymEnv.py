import gym, random

class GymEnv:
    #
    # Convert Gym environment into a multi-agent environment
    #
    
    def __init__(self, env, tlimit=None):
        if isinstance(env, str):
            env = gym.make(env)
        self.TLimit = tlimit
        self.Env = env
        self.Obs = None
        self.Reward = None
        self.Done = False
        self.Info = None
        self.Agents = []
        self.ValidActions = range(self.Env.action_space.n)
        
    def __str__(self):
        return "GymEnv(%s)" % (self.Env,)
        
    def randomMoves(self, agents):
        assert len(agents) == 1
        state0, action, state1, reward, done, info = self.Env.randomStep()
        if self.TLimit:
            done = done or random.random() * self.TLimit+1 < 1.0
        return [(agents[0], state0, action, state1, reward, done, self.ValidActions, info)]
        
    def reset(self, agents, random_placement = False):
        assert len(agents) == 1
        self.Done = False
        self.Obs = self.Env.reset()
        self.T = self.TLimit

    def addAgent(self, agent, random = False):
        pass
        
    def observe(self, agents):
        assert len(agents) == 1
        return [(agents[0], self.Obs, self.ValidActions, self.Done, self.Info)]
        
    def step(self, actions):
        self.Obs, self.Reward, self.Done, self.Info = self.Env.step(actions[0][1])
        if self.T is not None:
            self.T -= 1
            self.Done = self.Done or self.T <= 0
        
    def feedback(self, agents):
        return [(agents[0], self.Reward, self.Info)]
        
    def __getattr__(self, name):
        return getattr(self.Env, name)
        
    