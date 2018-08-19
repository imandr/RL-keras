import numpy as np
import math, random

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
import keras.backend as K

from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta

class QNet(object):
    
    RANGE = (-1.0, 1.0)
    DELTA = 0.1
    
    def __init__(self, R, r0, r1, G):
        self.R = R
        self.R0 = r0
        self.R1 = r1
        self.G = G
        self.QModel = self.create_model(1, 3)
        self.TrainModel = self.create_train_model(self.QModel)
        
    def q(self, xvector):
        return self.QModel.predict(xvector[:,None])

    def create_model(self, nin, nout):
    
        inp = Input(shape=(nin,), name="x")
        h1 = Dense(20, activation="tanh")(inp)
        h2 = Dense(80, activation="softplus")(h1)
        h3 = Dense(30, activation="softplus")(h2)
        out = Dense(nout, activation="linear")(h3)
    
        return Model(inp, out)
    
    def create_train_model(self, model):
    
        def differential(args):
            q0, q1, final, mask = args
            q0 = K.sum(q0*mask, axis=-1)[:,None]
            q1max = K.max(q1, axis=-1)[:,None]
            diff = q0 - (1.0-final) * self.G * q1max
            return diff
    
        x_shape = model.inputs[0].shape[1:]
        q_shape = model.output.shape[1:]
        x0 = Input(name="observation0", shape=x_shape)
        q0 = model(x0)
        x1 = Input(name="observation1", shape=x_shape)
        q1 = model(x1)
        mask = Input(name='mask', shape=q_shape)
        final = Input(name="final", shape=(1,))
        
        reward = Lambda(differential, name="reward")([q0, q1, final, mask])
        
        trainable = Model(inputs = [x0, x1, final, mask], outputs = reward)
    
        trainable.compile(
                optimizer=Adadelta(), 
                metrics=["mse"],      # metrics for the second output
                loss='mean_squared_error'
        )   
        return trainable
        
    def train_minibatch(self, mbsize):
        
        x0 = np.random.random((mbsize,))*(self.RANGE[1]-self.RANGE[0])+self.RANGE[0]
        r = np.random.random((mbsize,))

        a = np.ones((mbsize,), dtype=np.int32)
        a[r>=0.66] = 2
        a[r<0.33] = 0
        
        x1 = x0 + np.asarray(a-1, dtype=np.float32)*self.DELTA
        
        out_of_range = (x1 > self.RANGE[1]) + (x1 < self.RANGE[0])
        
        rewards = np.zeros((mbsize,))
        rewards[a==1] = self.R0
        rewards[a!=1] = self.R1
        rewards[out_of_range] = self.R

        inx = np.arange(mbsize)
        mask = np.zeros((mbsize,3))
        mask[inx,a] = 1.0
        final = np.zeros((mbsize,))
        final[out_of_range] = 1.0
        
        #print "batch:"
        #for xx0, xa, xx1, xmask, xr, xf in zip(x0, a, x1, mask, rewards, final):
        #    print xx0, xa, xx1, xmask, xr, xf
        
        metrics = self.TrainModel.train_on_batch([x0[:,None], x1[:,None], final[:,None], mask], rewards[:,None])
        return math.sqrt(metrics[0])



qn = QNet(-1.0, 0.0, 0.0, 0.8)

for epoch in xrange(100):
    for _ in range(1000):
        metrics = qn.train_minibatch(20)
        
    N = 20
    dx = (qn.RANGE[1]-qn.RANGE[0])/(N-1)
    x = qn.RANGE[0] + np.arange(N)*dx
    print "Metrics:", metrics, "   Q:"
    qvals = qn.q(x)
    for xi, qi in zip(x,qvals):
        print "%7.4f %s" % (xi, qi)
