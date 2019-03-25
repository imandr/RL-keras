import gym, random

class GymEnv:
    #
    # Convert 1-agent Gym environment into a multi-agent environment
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
        self.NAgents = 1
        
    def __str__(self):
        return "GymEnv(%s)" % (self.Env,)
        
    def randomMoves(self, size):
        moves = []
        for _ in xrange(size):
            o0 = self.Env.reset(random=True)
            action = env.action_space.sample()
            o1, reward, done, info = self.Env.step(action)
            moves.append((o0, action, o1, reward, done, info))
        return moves
        
    def reset(self, agents, random_placement = False):
        assert len(agents) == 1, "Converted Gym environments can not handle multiple agents"
        self.Done = False
        self.Obs = self.Env.reset()
        self.T = self.TLimit

    def addAgent(self, agent, random = False):
        raise ValueError("Converted Gym environments can not handle multiple agents")
        
    def observe(self, agents):
        assert len(agents) == 1, "Converted Gym environments can not handle multiple agents"
        return [(agents[0], self.Obs, self.ValidActions, self.Done, self.Info)]
        
    def step(self, actions):
        assert len(actions) == 1, "Converted Gym environments can not handle multiple agents"
        self.Obs, self.Reward, self.Done, self.Info = self.Env.step(actions[0][1])
        if self.T is not None:
            self.T -= 1
            self.Done = self.Done or self.T <= 0
        
    def feedback(self, agents):
        assert len(agents) == 1, "Converted Gym environments can not handle multiple agents"
        return [(agents[0], self.Obs, self.Reward, self.Done, self.Info)]
        
    def __getattr__(self, name):
        return getattr(self.Env, name)
        
    