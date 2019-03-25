import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
import keras.backend as K


def qmodel(inp_width, out_width):
    
    inp = Input((inp_width,))
    
    dense1 = Dense(inp_width*40, activation="tanh")(inp)
    dense2 = Dense(out_width*40, activation="softplus")(dense1)
    
    out = Dense(out_width, activation="linear")(dense2)
    
    model=Model(inputs=[inp], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model
    
class RLModel(object):
    
    def __init__(self, qmodel, gamma, *params, **args):
        self.QModel = qmodel
        self.TModel = self.trainig_model(qmodel, gamma, *params, **args)
        
    def fit_generator(self, generator, *params, **args):
        return self.TModel.fit_generator(
                (self.training_data(*data) for data in generator), 
                *params, **args
        )
        
    def predict_on_batch(self, *params, **args):
        return self.QModel.predict_on_batch(*params, **args)
        
    def __call__(self, *params, **kv):
        return self.QModel(*params, **kv)
        
    def training_data(self, *params):
        return params
        
    def trainig_model(self, qmodel, gamma, *params, **args):
        raise NotImplementedError
