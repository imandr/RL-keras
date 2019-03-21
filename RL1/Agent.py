

class Agent:
    
    def __init__(self, brain):
        self.Brain = brain
        self.Done = False
        self.Observation = None
        self.ActionSuggested = None
        self.ActionTaken = None
        self.QVector = None
        self.Trajectory = []
        
        self._EnvContext = None
        
    def init(self, observation):
        self.Trajectory = []
        self.Observation0 = observation
        self.Brain.episodeBegin()
    
    def action(self):
        a, qvector = self.Brain.action(self.Observation)
        self.ActionSuggested = a
        self.QVector = qvector
        return a
                
    def step(self, observation, action, reward, done):
        self.ActionTaken = action
        self.Trajectory.append((self.Observation, action, observation, reward, done))
        self.Observation = observation
        self.Done = done

    def end(self):
        self.Brain.episodeEnd()
        return self.Trajectory

    def info(self):
        return dict(
            "action_suggested": self.ActionSuggested,
            "qvector":          self.QVector
        )

    def trajectory(self, clear=False):
        t = self.Trajectory
        if clear:
            self.Trajectory = []
        return t
            
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
    