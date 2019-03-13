import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
import keras.backend as K

def qmodel(inp_width, out_width):
    
    inp = Input((inp_width,))
    
    dense1 = Dense(inp_width*40, activation="tanh")(inp)
    dense2 = Dense(out_width*40, activation="tanh")(dense1)
    
    out = Dense(out_width, activation="linear")(dense2)
    
    model=Model(inputs=[inp], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model
    
def training_model(qmodel, weight, gamma):

    x_shape = qmodel.inputs[0].shape[1:]
    #print "x_shape=", x_shape
    q_shape = qmodel.output.shape[1:]


    final = Input(shape=(1,))
    mask = Input(shape=q_shape)
    s0 = Input(shape=x_shape)
    s1 = Input(shape=x_shape)
    
    #
    # Q(s0,i) = r + gamma * max_j(Q(s1,j))
    # Q(s0,i) - gamma * max_j(Q(s1,j)) -> r
    #
    
    q0 = qmodel(s0)
    q1 = qmodel(s1)
    
    def differential(args):
        q0, q1, final, mask = args
        q0 = K.sum(q0*mask, axis=-1)[:,None]
        q1max = K.max(q1, axis=-1)[:,None]
        reward = q0 - (1.0-final) * gamma * q1max
        return reward
    
    out = Lambda(differential)([q0, q1, final, mask])
    model = Model(inputs=[s0, s1, mask, final], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model