import numpy as np
import math

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import QNet, QBrain
from RLKeras.multi import MultiDQNAgent, SynchronousMultiAgentController
from RLKeras.callbacks import Callback, Visualizer
from RLKeras.policies import GreedyEpsPolicy, BoltzmannQPolicy

from trade_env import MarketEnv, Stock


class TradeAgent(MultiDQNAgent):
    
    def __init__(self, env):
        model = self.create_model(env.observation_space.shape[0], env.action_space.n)
        brain = QBrain(model, kind="diff", v_selectivity=False, gamma=0.8)
        brain.compile(Adam(lr=1e-3), ["mse"])
        MultiDQNAgent.__init__(self, env, brain, train_sample_size=2000, train_batch_size=20)
        
    def create_model(self, inp_width, out_width):
        inp = Input((inp_width,))
        dense1 = Dense(300, activation="tanh", name="dense1")(inp)
        dense2 = Dense(100, activation="relu", name="dense2")(dense1)
        dense3 = Dense(10, activation="relu", name="dense3")(dense2)
        out = Dense(out_width, activation="linear", name="out_linear")(dense3)
        model = Model(inp, out)
        print("--- model summary ---")
        print(model.summary())
        return model
        

# Get the environment and extract the number of actions.

