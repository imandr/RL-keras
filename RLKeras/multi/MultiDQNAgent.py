from ..policies import GreedyEpsPolicy
import numpy as np
import random

class MultiDQNAgent:
    
    def __init__(self, env, qbrain, 
            callbacks = None,
            test_policy = None, train_policy = None,
            train_sample_size = 1000, train_rounds = 1, train_batch_size = 30,
            trains_between_updates = None):
        self.Env = env
        self.Brain = qbrain
        self.Callbacks = callbacks

        self.Done = False
        self.Observation0 = None
        self.Observation1 = None
        self.Valids1 = None
        self.Action0 = None
        self.Action1 = None
        self.Reward0 = None
        self.Reward1 = None

        self.TrainPolicy = train_policy or GreedyEpsPolicy(0.3)
        self.TestPolicy = test_policy or GreedyEpsPolicy(0.0)
        
        self.TrainSampleSize = train_sample_size
        self.TrainBatchSize = train_batch_size
        self.TrainsBetweenUpdates = trains_between_updates
        self.TrainsToUpdate = trains_between_updates
        self.TrainRoundsPerSession = train_rounds

        self.SessionsTrained = 0
        self.BatchesTrained = 0
        self.SamplesTrained = 0
        self.BrainUpdated = 0
        #self.StepsToWarmup = steps_to_warmup

    def action(self, observation, valid_actions, training, policy=None):
        self.Observation0 = self.Observation1
        self.Observation1 = observation
        self.Valids1 = valid_actions
        
        qvector = self.Brain.qvector(observation)
        self.QVector = qvector
        policy = policy or (self.TrainPolicy if training else self.TestPolicy)
        #print "using policy", policy
        action = policy(qvector, valid_actions)
        #print "t:", training,"  o:", observation, "  q:",qvector, "  a:", action,"(",np.argmax(qvector),")"
        #print "action=", action
        return action
                
    def learn(self, action, reward):
        self.Action0 = self.Action1
        self.Action1 = action
        self.Reward0 = self.Reward1
        self.Reward1 = reward
        if self.Observation0 is not None:
            if self.Reward0 is None:
                raise ValueError("action0, action1, reward0, reward1=%s,%s,%s,%s" % (self.Action0, self.Action1, self.Reward0, self.Reward1))
            self.Brain.memorize((self.Observation0, self.Action0, self.Reward0, 
                self.Observation1, False, self.Valids1), self.Reward0)    

    def trainBrain(self, callbacks):
        metrics = None
        if self.Brain.recordSize() >= self.TrainBatchSize:
            for train_round in xrange(self.TrainRoundsPerSession):
                metrics = self.Brain.train(self.TrainSampleSize, self.TrainBatchSize)
                self.LastMetrics = metrics
                self.SamplesTrained += self.TrainSampleSize
                self.BatchesTrained += 1
                #print "metrics:"
                #for m, mn in zip(metrics, metrics_names):
                #    print "   %s: %s" % (mn, m)
                if callbacks is not None:
                    callbacks.on_train_batch_end(self.BatchesTrained,
                        {
                            "train_sessions": self.SessionsTrained,
                            "train_samples": self.SamplesTrained,
                            "metrics": metrics,
                            "train_batches": self.BatchesTrained,
                            "memory_size": self.Brain.recordSize()
                        })
            
            self.SessionsTrained += 1
            #print "trainBran: callbacks:", callbacks
            if callbacks is not None:
                callbacks.on_train_session_end(self.SessionsTrained,
                    {
                        "train_sessions": self.SessionsTrained,
                        "train_samples": self.SamplesTrained,
                        "metrics": metrics,
                        "train_batches": self.BatchesTrained,
                        "memory_size": self.Brain.recordSize()
                    })
            if self.TrainsBetweenUpdates is not None:
                self.TrainsToUpdate -= 1
                if  self.TrainsToUpdate <= 0:
                    self.Brain.update()
                    self.TrainsToUpdate = self.TrainsBetweenUpdates
        return metrics

            
    def episodeBegin(self):
        self.Valids1 = None
        self.Action0 = None
        self.Action1 = None
        self.Reward0 = None
        self.Reward1 = None
        self.Done = False
        self.Observation0 = None
        self.Observation1 = None
        
    def episodeEnd(self, observation):
        pass
    
    def final(self, observation, training):
        if training:
            self.Brain.memorize((self.Observation1, self.Action1, self.Reward1, 
                observation, True, []), self.Reward1)   
        

