import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam

from models import DirectDiffModel

def defaultQModel(inp_width, out_width):
    
    inp = Input((inp_width,))
    
    dense1 = Dense(inp_width*10, activation="softplus")(inp)
    dense2 = Dense(out_width*10, activation="softplus")(dense1)
    
    out = Dense(out_width, activation="linear")(dense2)
    
    model=Model(inputs=[inp], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model
    
class Brain(object):
    
    def __init__(self, qmodel, model_type, policy, gamma, *params, **args):
        self.QModel = qmodel
        self.RLModel = self.create_rl_model(qmodel, model_type, gamma, *params, **args)
        self.Policy = policy
        self.TModel = self.RLModel.tmodel()
        
    def create_rl_model(self, qmodel, model_type, gamma, *params, **args):
        if model_type == "ddiff":
            return DirectDiffModel(qmodel, gamma, *params, **args)
        
    def tmodel(self):
        return self.TModel
        
    def q(self, obsrvation):
        return self.QModel.predict_on_batch([observation])[0]
        
    def action(self, observation):
        q = self.q(observation)
        a = self.Policy(q)
        return a, q

    def trainingData(self, o0, a, o1, r, f):
        return self.RLModel.training_data(o0, a, o1, r, f)

class Trainer(object):
    def __init__(self, env, brain, memory_size, random_mix, mbsize):
        source = MixedDriver(mbsize, env, brain, random_mix)
        self.Memory = ReplayMemory(source, memory_size)
        self.MBSize = mbsize
        
    def __iter__(self):
        return self.Memory.generate_samples(self.MBSize)

        
