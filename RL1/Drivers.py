class Driver(object):
    
    def __init__(self, env):
        self.Env = env
    
    def sample(self, n):
        raise NotImplementedError

class RandomDriver(Driver):
    
    def __init__(self, env):
        Driver.__init__(self, env)
    
    def sample(self, size):
        return self.Env.randomMoves(size)
        
class GameDriver(Driver):
    
    def __init__(self, env, nagents, agent_class, brain, *params):
        Driver.__init__(self, env)
        self.Agents = [agent_class(brain, *params) for _ in xrange(nagents)]
    
    def sample(self, size):
        samples = []
        while len(samples) < size:
            observations = self.Env.init(self.Agents)
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
            
            
            
         
    
    