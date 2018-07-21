import gym

class GymEnv:
    #
    # Convert Gym environment into a multi-agent environment
    #
    
    def __init__(self, env_name):
        self.Env = gym.make(env_name)
        self.Obs = None
        self.Reward = None
        self.Done = False
        self.Info = None
        self.Agents = []
        self.ValidActions = range(self.Env.action_space.n)
        
    def reset(self, agents, random_placement = False):
        assert len(agents) == 1
        self.Done = False
        self.Obs = self.Env.reset()

    def addAgent(self, agent, random = False):
        pass
        
    def observe(self, agents):
        assert len(agents) == 1
        return [(agents[0], self.Obs, self.ValidActions, self.Done, self.Info)]
        
    def step(self, actions):
        self.Obs, self.Reward, self.Done, self.Info = self.Env.step(actions[0][1])
        
    def feedback(self, agents):
        return [(agents[0], self.Reward, self.Info)]
        
    def __getattr__(self, name):
        return getattr(self.Env, name)