import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta

def function_model(inp_width, out_width):
    
    inp = Input((inp_width,))
    
    dense1 = Dense(inp_width*20, activation="tanh")(inp)
    dense2 = Dense(out_width*20, activation="softplus")(dense1)
    
    out = Dense(out_width, activation="linear")(dense2)
    
    model=Model(inputs=[inp], outputs=[out])
    #model.compile(Adam(lr=1e-3), ["mse"])
    return model
    
def training_model(inp_width, function_model):
    boundary = Input((1,))
    s0 = Input((inp_width,))
    s1 = Input((inp_width,))
    z0 = function_model(s0)
    z1 = function_model(s1)
    
    def combine_function(args):
        z0, z1, boundary = args
        return z1 - (1.0-boundary) * z0
    
    out = Lambda(combine_function)([z0, z1, boundary])
    model = Model(inputs=[s0, s1, boundary], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model