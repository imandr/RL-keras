import numpy as np
import math

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad, Adadelta

from RLKeras import QBrain
from RLKeras.multi import MultiDQNAgent
#from RLKeras.callbacks import Callback, Visualizer
#from RLKeras.policies import GreedyEpsPolicy, BoltzmannQPolicy


class CartPoleAgent(MultiDQNAgent):
    
    def __init__(self, env, kind="diff", gamma=0.8, weight=0.9, advantage=False):
        if kind == "qv":
            qmodel, vmodel = self.create_qv_models(env.observation_space.shape[0], env.action_space.n)
            model = (qmodel, vmodel)
        else:
            model = self.create_model(env.observation_space.shape[0], env.action_space.n)
            
        brain = QBrain(model, 
            kind=kind, advantage=advantage,
            gamma=gamma, v_selectivity=False, 
            qnet_soft_update=0.01, diff_qnet_weight=weight)
        brain.compile(Adadelta(), ["mse"])
        MultiDQNAgent.__init__(self, env, brain, train_sample_size=1000, train_batch_size=50)
        
    def create_model(self, inp_width, out_width):
        inp = Input((inp_width,))
        dense1 = Dense(30, activation="tanh", name="dense1")(inp)
        dense2 = Dense(60, activation="relu", name="dense2")(dense1)
        dense3 = Dense(20, activation="relu", name="dense3")(dense2)
        out = Dense(out_width, activation="linear", name="out_linear")(dense3)
        model = Model(inp, out)
        print("--- model summary ---")
        print(model.summary())
        return model
        
    def create_qv_models(self, inp_width, out_width):
        inp = Input((inp_width,))
        dense1 = Dense(30, activation="tanh", name="dense1")(inp)
        common = Dense(60, activation="relu", name="common")(dense1)
        
        # Q branch
        q1 = Dense(20, activation="relu", name="q1")(common)
        q2 = Dense(20, activation="relu", name="q2")(q1)
        qout = Dense(out_width, activation="linear", name="out_linear")(q2)
        
        # V branch
        v1 = Dense(20, activation="relu", name="v1")(common)
        v2 = Dense(20, activation="relu", name="v2")(v1)
        vout = Dense(1, activation="linear", name="out_linear")(v2)

        qmodel = Model(inp, qout)
        vmodel = Model(inp, vout)
        print("--- Q model summary ---")
        print(qmodel.summary())
        print("--- V model summary ---")
        print(vmodel.summary())
        return qmodel, vmodel
        
        
        

# Get the environment and extract the number of actions.

