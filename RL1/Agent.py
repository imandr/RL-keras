class Agent:
    
    def __init__(self, env, brain):
        self.Env = env
        self.Brain = brain
        self.Done = False
        self.Observation = None
        self.ActionSuggested = None
        self.ActionTaken = None
        self.QVector = None
        self.Trajectory = []
        
        self._State = None
        
    def init(self, observation):
        self.Trajectory = []
        self.Observation0 = observation
        return self.Brain.episodeBegin()
    
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
        info = self.Brain.episodeEnd()
        return self.Trajectory, info

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
            
