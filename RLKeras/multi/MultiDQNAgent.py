import numpy as np
import random

class MultiDQNAgent:
    
    def __init__(self, env, qbrain, 
            callbacks = None,
            train_sample_size = 1000, train_rounds = 1, train_batch_size = 20):
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

        self.TrainSampleSize = train_sample_size
        self.TrainBatchSize = train_batch_size
        self.TrainRoundsPerSession = train_rounds

        self.SessionsTrained = 0
        self.BatchesTrained = 0
        self.SamplesTrained = 0
        self.BrainUpdated = 0
        #self.StepsToWarmup = steps_to_warmup
        
    def memorize(self, tup, weight):
        self.Brain.memorize(tup, weight)

    def episodeBegin(self):
        #print "MultiDQNAgent: episodeBegin()"
        self.Valids1 = None
        self.Action0 = None
        self.Action1 = None
        self.Reward0 = None
        self.Reward1 = None
        self.Done = False
        self.Observation0 = None
        self.Observation1 = None
        
    def action(self, observation, valid_actions, training, policy):
        #print "MultiDQNAgent: action()"
        self.Observation0 = self.Observation1
        self.Observation1 = observation
        self.Valids1 = valid_actions
        
        qvector = self.Brain.qvector(observation)
        self.QVector = qvector
        #print "using policy", policy
        action = policy(qvector, valid_actions)
        #if policy.tau == 0.0001:
        #    print "qvector:", qvector, "   action:", action
        #print "t:", training,"  o:", observation, "  q:",qvector, "  a:", action,"(",np.argmax(qvector),")"
        #print "action=", action
        return action
                
    def learn(self, action, reward):
        #print "MultiDQNAgent: learn()"
        self.Action0 = self.Action1
        self.Action1 = action
        self.Reward0 = self.Reward1
        self.Reward1 = reward
        if self.Observation0 is not None:
            if self.Reward0 is None:
                raise ValueError("action0, action1, reward0, reward1=%s,%s,%s,%s" % (self.Action0, self.Action1, self.Reward0, self.Reward1))
            self.Brain.memorize((self.Observation0, self.Action0, self.Reward0, 
                self.Observation1, False, self.Valids1), self.Reward0)    

    def final(self, observation, training):
        #print "MultiDQNAgent: final()"
        if training:
            self.Brain.memorize((self.Observation1, self.Action1, self.Reward1, 
                observation, True, []), self.Reward1)   

    def episodeEnd(self, observation):
        pass

    def trainBrain(self, callbacks):
        metrics = None
        if self.Brain.recordSize() >= self.TrainBatchSize:
            nmetrics = 0
            summetrics = 0.0
            for train_round in xrange(self.TrainRoundsPerSession):
                metrics = self.Brain.train(self.TrainSampleSize, self.TrainBatchSize)
                self.SamplesTrained += self.TrainSampleSize
                self.BatchesTrained += 1

                nmetrics += 1
                summetrics += metrics
                
                #print "metrics:", metrics
                
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
                        "mean_metrics": None if nmetrics == 0 else summetrics/nmetrics,
                        "train_batches": self.BatchesTrained,
                        "memory_size": self.Brain.recordSize()
                    })
        return metrics

            
    
        

