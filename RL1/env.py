class Env:
    
    NAgents = 2
    
    def reset(self, agents, random = False):
        return observations
        
    def addAgent(self, agent, random = False):
        return observation
            
    def step(self, actions):
        # either single agent and action
        # or list of actions for all the agents, including those "done".Those Done will be ignored
        
        for agent, action in actions:
            apply_action(agent, action)
        
        return [info]
        
    def feedback(self, agents):
        if not isinstance(agents, list):
            agents = [agents]
        return [(agent, new_observation, reward, done, info) for agent in agents]
        
    def randomMoves(self, size):
        state0, action, state1, reward, done, info = self.Env.randomStep()
        if self.TLimit:
            done = done or random.random() * self.TLimit+1 < 1.0
        return (state0, action, state1, reward, done, info)
    