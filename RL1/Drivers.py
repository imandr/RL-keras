class Driver(object):
    
    def __init__(self, env):
        self.Env = env
    
    def sample(self, n):
        raise NotImplementedError

class RandomDriver(Driver):
    
    def __init__(self, env):
        Driver.__init__(self, env)
    
    def samples(self, size):
        return self.Env.randomMoves(size)
        
class GameDriver(Driver):
    
    def __init__(self, env, brain, nagents = None):
        Driver.__init__(self, env)
        self.Agents = [Agent(env, brain) for _ in xrange(nagents or env.NAgents)]
    
    def samples(self, size):
        samples = []
        while len(samples) < size:
            observations = self.Env.reset(self.Agents)
            active_agents = self.Agents
            for agent, observation in zip(active_agents, observations):
                agent.init(observation)
            while len(active_agents):
                agent_actions = [(a, a.action()) for a in active_agents]
                infos = self.Env.step(agent_actions)
                feedback = self.Env.feedback(active_agents)
                active_agents = []
                for agent, new_observation, reward, done, info in feedback:
                    agent.step(new_observation, reward, done)
                    if done:
                        samples += agent.trajectory(clear=True)
                    else:
                        active_agents.append(agent)
        return samples
        
class MixedDriver(Driver):
    def __init__(self, chunk, env, brain, random_fraction = 0.0):
        self.RandomFraction = random_fraction
        self.NGeneratedRandom = 0
        self.NGeneratedGame = 0
        self.Env = env
        self.Brain = brain
        self.GameDriver = GameDriver(env, brain)
        self.RandomDriver = RandomDriver(env)
        self.ChunkSize = chunk
        
    def chunk(self):
        generate_random = self.RandomFraction > 0.0
        ntotal = self.NGeneratedRandom + self.NGeneratedGame
        if ntotal > 0 and generate_random:
            current_fraction = float(self.NGeneratedRandom)/float(ntotal)
            generate_random = current_fraction < self.RandomFraction
        if generate_random:
            samples = self.RandomDriver.samples(self.ChunkSize)
            self.NGeneratedRandom += len(samples)
        else 
            samples = self.GameDriver.samples(self.ChunkSize)
            self.NGeneratedGame += len(samples)
        return samples
        
    def samples(self, size):
        s = []
        while len(s) < size:
            s += self.chunk()
        return s
        
        
        
        
        
    
    