import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
import keras.backend as K

from model import RLModel
        
class DirectDiffModel(RLModel):
    
    def __init__(self, qmodel, gamma, *params, **args):
        RLModel.__init__(self, qmodel, gamma, *params, **args)
        self.NActions = qmodel.outputs[0].shape[-1]
        
    def create_tmodel(self, qmodel, gamma):
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
        
    def training_data(self, s0, action, s1, final, reward):
        n_actions = self.NActions
        mask = np.zeros((len(s0), n_actions))
        for i in xrange(n_actions):
            mask[action==i, i] = 1.0
        return [s0, mask, s1, final], [reward]

