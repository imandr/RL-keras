import numpy as np
from keras.models import model_from_config, Model
from keras.layers import Lambda, Input, Layer, Dense
import keras.backend as K
from keras import optimizers
from tools import best_valid_action, format_batch
import time, math

class DifferentialQNet(object):
    def __init__(self, model, gamma = 0.9):
        self.Model = model
        self.TrainSamples = 0
        self.Gamma = K.variable(gamma, dtype='float32')
        self.NActions = self.Model.output.shape[-1]
        
    def get_weights(self):
        return self.Model.get_weights()
        
    def set_weights(self, weights):
        self.Model.set_weights(weights)

    def blend_weights(self, alpha, weights):
        my_weights = self.get_weights()
        x = my_weights[0].flat[0]
        assert len(weights) == len(my_weights)
        for my_w, w in zip(my_weights, weights):
            #print "blend_weights:", my_w.flat[:5], w.flat[:5]
            my_w.flat[:] = (my_w + alpha*(w-my_w)).flat
            #print my_w.flat[:10]
        self.set_weights(my_weights)
        #print "blend_weights:", x, "->", my_weights[0].flat[0], "->", self.Model.get_weights()[0].flat[0]
        
    def compile(self, optimizer, metrics=[]):
        
        self.Model.compile(optimizer='sgd', loss='mse')
        
        x_shape = self.Model.inputs[0].shape[1:]
        #print "x_shape=", x_shape
        q_shape = self.Model.output.shape[1:]
        #print "q_shape=", q_shape
        x0 = Input(name="observation0", shape=x_shape)
        q0 = self.Model(x0)
        x1 = Input(name="observation1", shape=x_shape)
        q1 = self.Model(x1)
        mask = Input(name='mask', shape=q_shape)
        final = Input(name="final", shape=(1,))
        
        def differential(args):
            q0, q1, final, mask = args
            q0 = K.sum(q0*mask, axis=-1)[:,None]
            q1max = K.max(q1, axis=-1)[:,None]
            diff = q0 - (1.0-final) * self.Gamma * q1max
            return diff
            
        reward = Lambda(differential, name="reward")([q0, q1, final, mask])
        
        trainable = Model(inputs = [x0, x1, final, mask], outputs = reward)
    
        trainable.compile(
                optimizer=optimizer, 
                metrics=metrics,      # metrics for the second output
                loss='mean_squared_error'
        )    

        print("--- trainable model summary ---")
        print(trainable.summary())
        
        self.TrainModel = trainable
        
    def compute(self, batch):
        return self.Model.predict_on_batch(batch)
        
    def train(self, sample, batch_size):
        # samples is list of tuples:
        # (last_observation, action, reward, new_observation, final, valid_actions, info)
        
        assert len(sample) >= batch_size
        
        for j in range(0, len(sample), batch_size):
            batches = zip(*sample[j:j+batch_size])
            batch_len = len(batches[0])
        
            state0_batch = format_batch(batches[0])
            action_batch = np.array(batches[1])
            reward_batch = np.array(batches[2])
            state1_batch = format_batch(np.array(batches[3]))
            final_state1_batch = batches[4]

            finals = np.asarray(final_state1_batch, dtype=np.int8).reshape((-1,1))
            inx_range = np.arange(batch_len, dtype=np.int32)
            masks = np.zeros((batch_len, self.NActions), dtype=np.float32)
            masks[inx_range, action_batch] = 1.0
            rewards = reward_batch.reshape((-1,1))
        
            t0 = time.time()
            
            #print finals.shape
        
            metrics = self.TrainModel.train_on_batch(
                [state0_batch, state1_batch, finals, masks], 
                rewards)
            self.TrainSamples += batch_len
        return metrics[0]
        
    def update(self):
        pass
        
