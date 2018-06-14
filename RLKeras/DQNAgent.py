from Agent import Agent
import numpy as np
import keras.backend as K
from keras import optimizers
from keras.models import Model, model_from_config
from keras.layers import Lambda, Input, Layer, Dense
from keras.losses import mean_squared_error
import random, math

class ReplayMemory:
    
    M = 1
    
    def __init__(self, size):
        self.MaxSize = size
        self.HighWater = int(size*1.3)
        self.Memory = []
        self.ShortTermMemory = []
        
    def add(self, tup):   
        self.ShortTermMemory.append(tup)
            
    def add_to_long(self, tups):
        self.Memory.extend(tups)
        if len(self.Memory) > self.HighWater:
            self.Memory = random.sample(self.Memory, self.MaxSize)
        
    def sample(self, size):
        shorts = self.ShortTermMemory[:size]
        n_short = len(shorts)
        self.ShortTermMemory = self.ShortTermMemory[n_short:]
        n_long = min(len(self.Memory), size - n_short)
        longs = random.sample(self.Memory, n_long) if n_long > 0 else []
        self.add_to_long(shorts)
        return shorts + longs
        
    def size(self):
        return len(self.Memory)+len(self.ShortTermMemory)

    def sizes(self):
        return len(self.Memory), len(self.ShortTermMemory)
        
def clone_model(model, custom_objects={}):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone
    
def max_valid(q, valid):
    return q[valid[np.argmax(q[valid])]]

def best_valid_action(q, valid):
    return valid[np.argmax(q[valid])]

def format_batch(batch):
    
    # if the model takes multiple inputs, then batch is a list of lists like [x1, x2, x3].
    # We need to convert that into [batch(x1), batch(x2, batch(x3)]
    
    if len(batch) == 0:
        return batch
        
    row = batch[0]
    if isinstance(row, (list, tuple)):
        return [np.array(x) for x in zip(*batch)]
    else:
        return np.array(batch)
        
class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, loss):
        updates = self.optimizer.get_updates(params=params, loss=loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()

def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates


class QNet(object):
    
    def __init__(self, model, soft_update = None):
        self.Model = model
        assert soft_update is None or isinstance(soft_update, float) and soft_update < 1.0
        self.SoftUpdate = soft_update

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
        
    def compute(self, batch):
        return self.Model.predict_on_batch(batch)
        
    def train(self, samples, gamma):
        # samples is list of tuples:
        # (last_observation, action, reward, new_observation, final, valid_actions, info)
        
        batch_size = len(samples)
        
        batches = zip(*samples)
        state0_batch = format_batch(batches[0])
        action_batch = np.array(batches[1])
        reward_batch = np.array(batches[2])
        state1_batch = format_batch(np.array(batches[3]))
        final_state1_batch = np.array(batches[4])
        final_mask_batch = 1.0 - final_state1_batch
        valid_actions_batch = np.array(batches[5])
        infos = batches[6]
        
        q_values = self.Model.predict_on_batch(state1_batch)
        nactions = q_values.shape[-1]
        actions = np.zeros((batch_size,), dtype=np.int32)
        for i, row in enumerate(q_values):
            valid = valid_actions_batch[i]
            assert final_state1_batch[i] or len(valid) > 0
            if len(valid):
                actions[i] = best_valid_action(row, valid)

        # Now, estimate Q values using the target network but select the values with the
        # highest Q value wrt to the online model (as computed above).
        target_q_values = self.TargetModel.predict_on_batch(state1_batch)
        qmax_state1_batch = target_q_values[range(batch_size), actions]

        #print "QNet.train: q_state1_batch:     ", q_state1_batch
        #print "QNet.train: valid_actions_batch:", valid_actions_batch
        #print "QNet.train: qmax_state1_batch:  ", qmax_state1_batch

        targets = np.zeros((batch_size, nactions))
        dummy_targets = np.zeros((batch_size,))
        masks = np.zeros((batch_size, nactions))

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
        # but only for the affected output units (as given by action_batch).
        # Set discounted reward to zero for all states that were terminal.
        #print "QNet.train:", gamma, qmax_state1_batch, final_mask_batch
        discounted_reward_batch = qmax_state1_batch * final_mask_batch * gamma
        
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
        metrics = self.TrainModel.train_on_batch([state0_batch, targets, masks], [dummy_targets, targets])
        metrics_names = self.TrainModel.metrics_names
        #print "-- metrics --"
        #for idx, (mn, m) in enumerate(zip(metrics_names, metrics)):
        #    print "%d: %s = %s" % (idx, mn, m)
        return metrics, metrics_names
        
    def update(self):
        self.TargetModel.set_weights(self.Model.get_weights())
        
        
        
            
class DQNAgent(Agent):
    
    def __init__(self, env, qnet, memory_size = 100000, 
            gamma = 0.9,
            steps_between_train = 100, episodes_between_train = 1, train_sample_size = 30, train_rounds = 20,
            trains_between_updates = 100,
            **super_args):
        
        Agent.__init__(self, env, **super_args)
        self.Memory = ReplayMemory(memory_size)
        self.QNet = qnet
        assert isinstance(qnet, QNet)
        self.reset_states()
        self.StepsBetweenTrain = steps_between_train
        self.EpisodesBetweenTrain = episodes_between_train
        self.StepsToTrain = steps_between_train
        self.EpisodesToTrain = episodes_between_train
        self.TrainSampleSize = train_sample_size
        self.TrainsBetweenUpdates = trains_between_updates
        self.TrainsToUpdate = trains_between_updates
        self.TrainRoundsPerSession = train_rounds
        self.SessionsTrained = 0
        self.BatchesTrained = 0
        self.SamplesTrained = 0
        self.QNetUpdated = 0
        #self.StepsToWarmup = steps_to_warmup
        self.Gamma = gamma
        
    def qvector(self, observation):
        x = format_batch([observation])
        #print "qvector: x=", x
        return self.QNet.compute(x)[0]
        
    def recordTransition(self, last_observation, action, reward, new_observation, final, valid_actions, info = None):
        if last_observation is not None:
            #print
            #print "     record: o0:", last_observation, " a:", action, \
            #    " r:",reward, " o1:", new_observation, " f:",final, " v:",valid_actions, " i:", info
            #print
            if final or random.random() < 0.5:
                tup = (last_observation, action, reward, new_observation, final, valid_actions, info)
                self.Memory.add(tup)

    def reset_states(self):
        self.LastObservation = None
        self.LastQVector = None
        self.LastAction = None
        self.LastReward = None

    def action(self, observation, policy):
        
        valid_actions = self.updateState(observation)
        qvector = self.qvector(observation)
        action = policy(qvector, valid_actions)
        
        #print "action: o0:", observation, " v:", valid_actions, " q:", qvector, " a:", action
        
        if self.Training:
            self.recordTransition(self.LastObservation, self.LastAction, self.LastReward, observation, False, valid_actions,
                info = {})

        self.LastObservation = observation
        self.LastQVector = qvector
        self.LastAction = action
        return action, valid_actions
        
    def learn(self, reward, new_observation, final):
        #print "learn: r:", reward, " o1:", new_observation, " f:", final
        metrics, metrics_names = None, None
        self.LastReward = reward
        self.StepsToTrain -= 1
        if self.StepsToTrain <= 0 or self.EpisodesToTrain <= 0:
            metrics, metrics_names = self.trainQNet()
        return metrics, metrics_names
            
    def final(self, final_observation):
        #print "final: o:", final_observation
        if self.Training:
            self.recordTransition(self.LastObservation, self.LastAction, self.LastReward, final_observation, True, [])
        
    
    def episodeEnd(self):
        if self.Training:
            self.EpisodesToTrain -= 1
        
    def trainQNet(self):
        metrics, metrics_names = None, None
        #print "trainQNet: memory sizes:", self.Memory.sizes()
        if self.Memory.size() >= self.TrainSampleSize:
            for train_round in xrange(self.TrainRoundsPerSession):
                samples = self.Memory.sample(self.TrainSampleSize)
                metrics, metrics_names = self.QNet.train(samples, self.Gamma)
                self.LastMetrics, self.LastMetricsNames = metrics, metrics_names
                self.SamplesTrained += len(samples)
                self.BatchesTrained += 1
                #print "metrics:"
                #for m, mn in zip(metrics, metrics_names):
                #    print "   %s: %s" % (mn, m)
                if self.Callbacks is not None:
                    self.Callbacks.on_train_batch_end(self.BatchesTrained,
                        {
                            "train_sessions": self.SessionsTrained,
                            "train_samples": self.SamplesTrained,
                            "metrics": metrics,
                            "metrics_names": metrics_names,
                            "train_batches": self.BatchesTrained,
                            "memory_size": self.Memory.size()
                        })
            self.StepsToTrain = self.StepsBetweenTrain
            self.EpisodesToTrain = self.EpisodesBetweenTrain
            
            self.SessionsTrained += 1

            if self.Callbacks is not None:
                self.Callbacks.on_train_session_end(self.SessionsTrained,
                    {
                        "train_sessions": self.SessionsTrained,
                        "train_samples": self.SamplesTrained,
                        "metrics": metrics,
                        "metrics_names": metrics_names,
                        "train_batches": self.BatchesTrained,
                        "memory_size": self.Memory.size()
                    })
            self.TrainsToUpdate -= 1
            if self.TrainsToUpdate <= 0 and self.QNet.SoftUpdate is None:
                self.updateQNet()
        return metrics, metrics_names
                
    def updateQNet(self):
        self.QNet.update()
        print "QNet upated"
        self.TrainsToUpdate = self.TrainsBetweenUpdates
        self.QNetUpdated += 1
        if self.Callbacks is not None:
            self.Callbacks.on_qnet_update(self.QNetUpdated, {})
        
        
            
        
        
    
