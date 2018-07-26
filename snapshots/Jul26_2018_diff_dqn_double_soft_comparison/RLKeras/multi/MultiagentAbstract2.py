

class Policy:
    
    def __call__(self, qvector):
        return action

class Agent:
    
    def __init__(self, brain):
        self.Done = False
        self.Observation0 = None
        self.Valids0 = None
        self.Observation1 = None
        self.Valids1 = None
        self.Action0 = None
        self.Action1 = None
        self.Reward0 = None
        self.Reward1 = None
        pass
        
    def episodeBegin(self):
        pass
    
    def action(self, observation, valid_actions, training, policy=None):
        # returns qvector
        self.Observation0 = self.Observation1
        self.Valids0 = self.Valids1
        self.Observation1 = observation
        self.Valids1 = valid_actions
        self.QVector = qvector
        return policy(qvector(observation))
                
    def learn(self, action, reward):
        self.Action0 = self.Action1
        self.Action1 = action
        self.Reward0 = self.Reward1
        self.Reward1 = reward
        if self.Observation0 is not None:
            record_memory(self.Observation0, self.Action0, self.Reward0, self.Observation1, False, self.Valids1)
        # train if needed
        return metrics
            
    def final(self, observation, training):
        if training:
            record_memory(self.Observation1, self.Action1, self.Reward1, observation, True, [])
        

class Env:
    
    def reset(self, agents, random = False):
        pass
        
    def addAgent(self, agent, random = False):
        pass
            
    def observe(self, agents):
        return [(agent, observation, valid_actions, done, info) for agent in agents]
    
    def step(self, actions):
        # either single agent and action
        # or list of actions for all the agents, including those "done".Those Done will be ignored
        
        for agent, action in actions:
            apply_action(agent, action)
        
        return [(agent, info) for agent in agents]
        
    def feedback(self, agents):
        if not isinstance(agents, list):
            agents = [agents]
        return [(agent, reward, info) for agent in agents]
        