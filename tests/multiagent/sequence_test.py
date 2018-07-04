import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad
from RLKeras import MultiDQNAgent, SynchronousMultiAgentController, QBrain

class SeqEnv(object):
    
    def __init__(self, t=5):
        self.TMax = t
        
    def reset(self, agents):
        self.T = 0
        self.R = -1
        
    def observe(self, agents):
        return [(agent, np.array([self.T]), [0], self.T >= self.TMax, {}) for agent in agents]
        
    def step(self, actions):
        self.T += 1
        self.R += 1
        return [(agent, {}) for agent, action in actions]
        
    def feedback(self, agents):
        return [(agent, self.R, {}) for agent in agents]
        
class SeqAgent(MultiDQNAgent):
    pass
    
def create_model():
    inp = Input((1,))
    dense1 = Dense(5, activation="relu", name="dense1")(inp)
    out = Dense(1, activation="linear", name="out_linear")(dense1)
    model = Model(inp, out)
    print("--- model summary ---")
    print(model.summary())
    return model

env = SeqEnv()
model = create_model()
brain = QBrain(model, soft_update=0.0001)
agent = SeqAgent(env, brain)
agents = [agent]
controller = SynchronousMultiAgentController(env, agents)
controller.fit(max_episodes = 1)

for tup in sorted(brain.Memory.ShortTermMemory):
    print tup

