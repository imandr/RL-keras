from Agent import Agent
import numpy as np
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense
from keras.losses import mean_squared_error
import random, math
from Memory import ReplayMemory
from QNet import QNet

from tools import max_valid, best_valid_action, format_batch
        

        
class DQNAgent(Agent):
    
    def __init__(self, env, qnet, memory, 
            gamma = 0.9,
            steps_between_train = 100, episodes_between_train = 1, train_sample_size = 30, train_rounds = 20,
            trains_between_updates = 100,
            **super_args):
        
        Agent.__init__(self, env, **super_args)
        self.Memory = memory        # ReplayMemory(memory_size), memory_size = 100000
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
            metrics = self.trainQNet()
        return metrics, metrics_names
            
    def final(self, final_observation):
        #print "final: o:", final_observation
        if self.Training:
            self.recordTransition(self.LastObservation, self.LastAction, self.LastReward, final_observation, True, [])
        
    
    def episodeEnd(self):
        if self.Training:
            self.EpisodesToTrain -= 1
        
    def trainQNet(self):
        print "trainQNet"
        metrics, metrics_names = None, None
        #print "trainQNet: memory sizes:", self.Memory.sizes()
        if self.Memory.size() >= self.TrainSampleSize:
            for train_round in xrange(self.TrainRoundsPerSession):
                samples = self.Memory.sample(self.TrainSampleSize)
                metrics = self.QNet.train(samples, self.Gamma)
                self.LastMetrics = metrics
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
                        "train_batches": self.BatchesTrained,
                        "memory_size": self.Memory.size()
                    })
            self.TrainsToUpdate -= 1
            if self.TrainsToUpdate <= 0 and self.QNet.SoftUpdate is None:
                self.updateQNet()
        return metrics
                
    def updateQNet(self):
        self.QNet.update()
        print "QNet upated"
        self.TrainsToUpdate = self.TrainsBetweenUpdates
        self.QNetUpdated += 1
        if self.Callbacks is not None:
            self.Callbacks.on_qnet_update(self.QNetUpdated, {})
        
        
            
        
        
    
