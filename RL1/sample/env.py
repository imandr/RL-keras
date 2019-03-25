class Env:
    
    Delta = 0.1
    NActions = 4
    StateDim = 2

    Moves = np.array(
        [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0)
        ]
    ) * Delta

    def randomStates(self, n):
        return np.random.random((n,2))
    
    def randomActions(self, n):
        return np.random.randint(0, self.NActions, n)

    def reset(self, agents, random=False):
        states = self.randomStates(len(agents))
        for a, s in zip(agents, states):
            agent._State = (s, 0.0)
        return states
        
    def step(self, agents_actions):
            n = len(agents_actions)
            done = np.zeros((n,), dtype=np.bool)
            rewards = np.zeros((n,))
            states1 = []
            for i, (agent, action) in enumerate(agents_actions):
                done[i] = agent.Done
                s0, _ = agent._State
                s1 = s0
                r = 0.0
                if not agent.Done:
                    s1 = agent._State + self.Moves[action]
                    x, y = s1
                    final = x > 1.0 or y > 1.0
                    if final:
                        z = x if y > 1.0 else y
                        r = 1-2*z
                        agent.Done = True
                agent._State = (s1, r)
                
    def feedback(self, agents):
        return [(agent, agent._State[0], agent._State[1], agent.Done, {}) for agent in agents]
    
        
        
    