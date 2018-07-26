import numpy as np
from keras.models import model_from_config, Model
from keras.layers import Lambda, Input, Layer, Dense
import keras.backend as K
from keras import optimizers
from tools import best_valid_action, format_batch
import time, math

def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone
    
class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, loss):
        updates = self.optimizer.get_updates(params=params, loss=loss)
        self.updates = updates + self.additional_updates
        return self.updates

    def get_updates___(self, params, loss):
        updates = self.optimizer.get_updates(params=params, loss=loss)
        updates += self.additional_updates
        self.updates = updates + self.additional_updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()

def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = [K.update(tw, tw + tau*(sw-tw)) for tw, sw in zip(target_weights, source_weights)]
    return updates

class DQN(object):
    
    #
    # Depending on kind, implements 3 different types of network:
    #
    # kind="naive": 
    # model fits to itself:
    #
    #    M(x0) -> reward + gamma * max(M(x1)) * (1-final)
    #
    # kind="dqn": 
    # Implements ideas from Mnih (2015):
    # Online model M fits to target model T:
    #
    #     M(x0) -> reward + gamma * max(T(x1)) * (1-final)
    #
    # and M is copied to T periodically
    #
    # kind="double"
    # Implememts Hasselt (2015):
    # Online model M fits to target model T:
    #
    #     M(x0) -> reward + gamma * (T(x1)[argmax(M(x1))]) * (1-final)
    # 
    # and M is copied to T periodically
    #
    
    def __init__(self, model, kind="dqn", hard_update_samples = 10000, gamma=0.99):
        self.Model = model
        self.TrainSamples = 0
        self.TrainSamplesBetweenUpdates = self.TrainSamplesToNextUpdate = hard_update_samples
        self.Gamma = gamma
        self.XWidth = self.Model.inputs[0].shape[-1]
        self.NActions = self.Model.output.shape[-1]
        self.Kind = kind
        self.SoftUpdate = 0.01

    def compile(self, optimizer, metrics=[]):
        
        self.TargetModel = clone_model(self.Model)
        self.TargetModel.compile(optimizer='sgd', loss='mse')
        self.Model.compile(optimizer='sgd', loss='mse')
        
        # build trainable model

        x_shape = (self.XWidth,)
        q_shape = (self.NActions,)

        x0 = Input(name="x0", shape=x_shape)
        mask = Input(name='mask', shape=q_shape)

        q0 = self.Model(x0)

        def masked_out(args):
            q0, mask = args
            return K.sum(q0*mask, axis=-1)[:,None]      # make it (n,1) shape
        
        out = Lambda(masked_out)([q0, mask])
        
        trainable = Model(inputs=[x0, mask], output=out)

        if self.SoftUpdate is not None:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.TargetModel, self.Model, self.SoftUpdate)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)
        
        print("--- trainable model summary ---")
        print(trainable.summary())
        
        trainable.compile(optimizer=optimizer, loss="mean_squared_error", metrics=metrics)
        self.TrainModel = trainable

    def get_weights(self):
        return (self.Model.get_weights(), self.TargetModel.get_weights())
        
    def blend_weights(self, alpha, weights):
        
        mw, tw = weights
        
        my_mw, my_tw = self.Model.get_weights(), self.TargetModel.get_weights()
        
        assert len(mw) == len(my_mw)
        for my_w, x_w in zip(my_mw, mv):
            my_w.flat[:] = my_w.flat + alpha*(x_w.flat-my_w.flat)

        assert len(tw) == len(my_tw)
        for my_w, x_w in zip(my_tw, tv):
            my_w.flat[:] = my_w.flat + alpha*(x_w.flat-my_w.flat)

        self.set_weights((my_mw, my_tw))
        
    def set_weights(self, weights):
        mw, tw = weights
        self.Model.set_weights(mw)
        self.TargetModel.set_weights(tw)

        
    def compute(self, batch):
        return self.Model.predict_on_batch(batch)
        
    def train(self, sample, batch_size):
        # samples is list of tuples:
        # (last_observation, action, reward, new_observation, final, valid_actions, info)
        
        #print "samples:"
        #for s in samples:
        #    print s

        metrics = None
        
        for j in range(0, len(sample), batch_size):
            batches = zip(*sample[j:j+batch_size])
            batch_len = len(batches[0])
        
            state0_batch = format_batch(batches[0])
            action_batch = np.array(batches[1])
            reward_batch = np.array(batches[2])
            state1_batch = format_batch(np.array(batches[3]))
            final_state1_batch = np.asarray(batches[4], dtype=np.float32)
            mask_batch = np.zeros((batch_len, self.NActions), dtype=np.float32)
            irange = np.arange(batch_len)
            mask_batch[irange, action_batch] = 1.0
            
            if self.Kind == "double":
                q1t_batch = self.TargetModel.predict_on_batch(state1_batch)
                q1m_batch = self.Model.predict_on_batch(state1_batch)
                inx = np.argmax(q1m_batch, axis=-1)
                q1_batch = q1t_batch[irange, inx]
            elif self.Kind == "dqn":
                q1t_batch = self.TargetModel.predict_on_batch(state1_batch)
                q1_batch = np.max(q1t_batch, axis=-1)
            else:   # naive
                q1m_batch = self.Model.predict_on_batch(state1_batch)
                q1_batch = np.max(q1m_batch, axis=-1)
                
            #print "q1:", q1_batch
            #print "reward:", reward_batch
            #print "final:", final_state1_batch
            #print "action:", action_batch
            #print "mask:", mask_batch
                
            q_ = (reward_batch + self.Gamma * (1.0-final_state1_batch) * q1_batch)[:,None]
            #print "q_:", q_
            
            metrics = self.TrainModel.train_on_batch([state0_batch, mask_batch], q_)

            if self.SoftUpdate is None:
                self.TrainSamples += batch_len
                self.TrainSamplesToNextUpdate -= batch_len
            
                if self.TrainSamplesToNextUpdate <= 0:
                    self.TargetModel.set_weights(self.Model.get_weights())
                    print "======== Target network updated ========"
                    self.TrainSamplesToNextUpdate = self.TrainSamplesBetweenUpdates
                    
        return metrics[0]
        

        
class DualQNet(object):
    
    def __init__(self, model, soft_update = None, hard_update_samples = 1000, gamma=0.99):
        self.Model = model
        assert soft_update is None or isinstance(soft_update, float) and soft_update < 1.0
        self.SoftUpdate = soft_update
        self.TrainSamples = 0
        if soft_update is not None: hard_update_samples = None
        self.TrainSamplesBetweenUpdates = self.TrainSamplesToNextUpdate = hard_update_samples
        self.Gamma = gamma

    def compile(self, optimizer, metrics=[]):
        
        self.TargetModel = clone_model(self.Model)
        self.TargetModel.compile(optimizer='sgd', loss='mse')
        self.Model.compile(optimizer='sgd', loss='mse')
        
        y_pred = self.Model.output
        out_shape = self.Model.output.shape[1:]
        y_true = Input(name='y_true_input', shape=out_shape)
        mask = Input(name='mask', shape=out_shape)

        def masked_error(args):
            y_true, y_pred, mask = args
            #assert y_true.shape == y_pred.shape, "y_true.shape %s != y_pred.shape %s" % (y_true.shape, y_pred.shape)
            #assert y_true.shape == mask.shape, "y_true.shape %s != mask.shape %s" % (y_true.shape, mask.shape)
            loss = K.square(y_true-y_pred)*mask
            return K.sum(loss, axis=-1)
        
        loss_out = Lambda(masked_error, output_shape=(1,), name='masked_loss')([y_true, y_pred, mask])
        ins = self.Model.inputs
        trainable = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        
        print("--- trainable model summary ---")
        print(trainable.summary())
        

        if self.SoftUpdate is not None:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.TargetModel, self.Model, self.SoftUpdate)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        #assert len(trainable_model.output_names) == 2
        #combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,                  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),    # we only include this for the metrics
        ]
        trainable.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        self.TrainModel = trainable

    def get_weights(self):
        return (self.Model.get_weights(), self.TargetModel.get_weights())
        
    def blend_weights(self, alpha, weights):
        
        mw, tw = weights
        
        my_mw, my_tw = self.Model.get_weights(), self.TargetModel.get_weights()
        
        assert len(mw) == len(my_mw)
        for my_w, x_w in zip(my_mw, mv):
            my_w.flat[:] = my_w.flat + alpha*(x_w.flat-my_w.flat)

        assert len(tw) == len(my_tw)
        for my_w, x_w in zip(my_tw, tv):
            my_w.flat[:] = my_w.flat + alpha*(x_w.flat-my_w.flat)

        self.set_weights((my_mw, my_tw))
        
    def set_weights(self, weights):
        mw, tw = weights
        self.Model.set_weights(mw)
        self.TargetModel.set_weights(tw)

        
    def compute(self, batch):
        return self.Model.predict_on_batch(batch)
        
    def train(self, sample, batch_size):
        # samples is list of tuples:
        # (last_observation, action, reward, new_observation, final, valid_actions, info)
        
        #print "samples:"
        #for s in samples:
        #    print s

        metrics = None
        
        for j in range(0, len(sample), batch_size):
            batches = zip(*sample[j:j+batch_size])
            batch_len = len(batches[0])
        
            state0_batch = format_batch(batches[0])
            action_batch = np.array(batches[1])
            reward_batch = np.array(batches[2])
            #print np.array(batches[3]).shape
            state1_batch = format_batch(np.array(batches[3]))
            final_state1_batch = np.array(batches[4])
        
            #for _, a, r, _, f, _ in samples:
            #    if f:
            #        print a, r, f
            final_mask_batch = 1.0 - final_state1_batch
            valid_actions_batch = np.array(batches[5])
            #infos = batches[6]
        
            q_values = self.Model.predict_on_batch(state1_batch)
            nactions = q_values.shape[-1]
            actions = np.zeros((batch_len,), dtype=np.int32)
            for i, row in enumerate(q_values):
                valid = valid_actions_batch[i]
                assert final_state1_batch[i] or len(valid) > 0
                if len(valid):
                    actions[i] = best_valid_action(row, valid)

            # Now, estimate Q values using the target network but select the values with the
            # highest Q value wrt to the online model (as computed above).
            target_q_values = self.TargetModel.predict_on_batch(state1_batch)
            qmax_state1_batch = target_q_values[range(batch_len), actions]

            #print "QNet.train: q_state1_batch:     ", q_state1_batch
            #print "QNet.train: valid_actions_batch:", valid_actions_batch
            #print "QNet.train: qmax_state1_batch:  ", qmax_state1_batch

            targets = np.zeros((batch_len, nactions))
            dummy_targets = np.zeros((batch_len,))
            masks = np.zeros((batch_len, nactions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            # Set discounted reward to zero for all states that were terminal.
            #print "QNet.train:", gamma, qmax_state1_batch, final_mask_batch
            discounted_reward_batch = qmax_state1_batch * final_mask_batch * self.Gamma
        
            #print
            #print "train: reward_batch=\n", reward_batch
            #print "train: final_mask_batch=\n", final_mask_batch
            #print "train: qmax_state1_batch=\n", qmax_state1_batch
            #print "train: discounted=\n", discounted_reward_batch
        
            #print discounted_reward_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            #print "train: reward_batch", reward_batch
            #print "train: discounted_reward_batch:", discounted_reward_batch
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            #print "train: target=\n", targets
            #print "train: masks=\n", masks

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            #print "QNet.train: y_true shapes:", dummy_targets.shape, targets.shape
            t0 = time.time()
            metrics = self.TrainModel.train_on_batch([state0_batch, targets, masks], [dummy_targets, targets])
            #print "train time:", time.time() - t0
            metrics_names = self.TrainModel.metrics_names
            #print "-- metrics --"
            #for idx, (mn, m) in enumerate(zip(metrics_names, metrics)):
            #    print "%d: %s = %s" % (idx, mn, m)
            self.TrainSamples += batch_len
            if self.TrainSamplesToNextUpdate is not None:
                self.TrainSamplesToNextUpdate -= batch_len
                if self.TrainSamplesToNextUpdate <= 0:
                    self.TargetModel.set_weights(self.Model.get_weights())
                    self.TrainSamplesToNextUpdate = self.TrainSamplesBetweenUpdates
                    
        return metrics[0]
        

        
