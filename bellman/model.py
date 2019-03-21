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
    
class CompoundModel(object):
    
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
        
class DirectDiffModel(CompoundModel):
    
    def __init__(self, qmodel, gamma, *params, **args):
        CompoundModel.__init__(self, qmodel, gamma, *params, **args)
        self.NActions = qmodel.outputs[0].shape[-1]
        
    def trainig_model(self, qmodel, gamma):
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
            q0, mask, q1, final = args
            q0 = K.sum(q0*mask, axis=-1)[:,None]
            q1max = K.max(q1, axis=-1)[:,None]
            reward = q0 - (1.0-final) * gamma * q1max
            return reward
    
        out = Lambda(differential)([q0, mask, q1, final])
        model = Model(inputs=[s0, mask, s1, final], outputs=[out])
        model.compile(Adam(lr=1e-3), ["mse"])
        return model
        
    def augment_data(self, x, r):
        n_actions = self.NActions
        s0, action, s1, final = x
        mask = np.zeros((len(s0), n_actions))
        for i in xrange(n_actions):
            mask[action==i, i] = 1.0
        return [s0, mask, s1, final], r

class LateralDiffModel(CompoundModel):
    
    def __init__(self, qmodel, gamma, weight):
        CompoundModel.__init__(self, qmodel, gamma, weight)
        self.NActions = qmodel.outputs[0].shape[-1]
        
    def trainig_model(self, qmodel, weight, gamma):

        x_shape = qmodel.inputs[0].shape[1:]
        #print "x_shape=", x_shape
        q_shape = qmodel.output.shape[1:]


        final = Input(shape=(1,))
        mask = Input(shape=q_shape)
        s0 = Input(shape=x_shape)
        q0i = Input(shape=q_shape)
        s1 = Input(shape=x_shape)
        q1i = Input(shape=q_shape)

        #
        # Q(s0,i) = r + gamma * max_j(Q(s1,j))
        # Q(s0,i) - gamma * max_j(Q(s1,j)) -> r
        #

        q0 = qmodel(s0)
        q1 = qmodel(s1)

        def differential(args):
            q0, mask, q1, final = args
            q0 = K.sum(q0*mask, axis=-1)[:,None]
            q1max = K.max(q1, axis=-1)[:,None]
            reward = q0 - (1.0-final) * gamma * q1max
            return reward
        
        def combine(args):
            r0, r1, final = args
            return (r0 + (1.0-final) * weight * r1)/(1.0 + weight * (1.0-final))

        r0 = Lambda(differential)([q0, mask, q1i, final])
        r1 = Lambda(differential)([q0i, mask, q1, final])
        out = Lambda(combine)([r0, r1, final])
        model = Model(inputs=[s0, q0i, mask, s1, q1i, final], outputs=[out])
        model.compile(Adam(lr=1e-3), ["mse"])
        return model
        
    def augment_data(self, x, r):
        n_actions = self.NActions
        s0, action, s1, final = x
        q0i = self.QModel.predict_on_batch(s0)
        q1i = self.QModel.predict_on_batch(s1)
        mask = np.zeros((len(s0), n_actions))
        for i in xrange(n_actions):
            mask[action==i, i] = 1.0
        return [s0, q0i, mask, s1, q1i, final], r

    
