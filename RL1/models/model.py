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
        
        def augment_data(g):
            for x, y in g:
                yield self.augment_data(x, y)
                
        return self.TModel.fit_generator(augment_data(generator), *params, **args)
        
    def predict_on_batch(self, s):
        return self.QModel.predict_on_batch(s)
        
    def __call__(self, *params, **kv):
        return self.QModel(*params, **kv)
        
    def augment_data(self, x, y):
        return x, y
        
    def trainig_model(self, qmodel, gamma, *params, **args):
        raise NotImplementedError
