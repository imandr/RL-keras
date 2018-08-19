from RLKeras.callbacks import Callback, Visualizer
from RLKeras.multi import SynchronousMultiAgentController
from RLKeras.policies import BoltzmannQPolicy
from RLKeras import GymEnv
from env import MountainCarEnv

from agent import Agent

import numpy as np
import math, getopt, sys

np.set_printoptions(precision=4, suppress=True)

class RunLogger(Callback):
    
    def __init__(self, csv_out = None, loss_info_from = None):
        if isinstance(csv_out, str):
            csv_out = open(csv_out, "w")
        if csv_out is not None:
            csv_out.write("episodes,rounds,steps,reward_per_episode,loss_ma\n")
        self.CSVOut = csv_out
        self.LossMA = None
        self.LossInfoFrom = loss_info_from
        
        
    def on_train_session_end(self, nsessions, logs):
        if logs["mean_metrics"] is not None: # training ?
            loss = math.sqrt(logs["mean_metrics"])
            if self.LossMA is None:
                self.LossMA = loss
            self.LossMA += 0.1 * (loss-self.LossMA)
    
    def on_run_end(self, param, logs={}):
        rewards = logs["run_rewards"]
        nagents = len(rewards)
        nepisodes = logs["nepisodes"]
        avg_reward = float(sum([r for t, r in rewards]))/nepisodes/nagents
        loss_ma = self.LossMA
        if self.LossInfoFrom is not None:
            loss_ma = self.LossInfoFrom.LossMA
        print "Run episodes/rounds/steps:", \
            logs["total_train_episodes"], \
            logs["total_train_rounds"], \
            logs["total_train_steps"], \
            "  Loss MA:%.6f" % (loss_ma,), \
            "  Session reward per episode:", avg_reward
        if self.CSVOut is not None:
            self.CSVOut.write("%d,%d,%d,%f,%f\n" % ( 
                    logs["total_train_episodes"],
                    logs["total_train_rounds"],
                    logs["total_train_steps"],
                    avg_reward, 
                    loss_ma if loss_ma is not None else -1.0
                )
            )
            self.CSVOut.flush()


class EpisodeLogger(Callback):
    
    def on_episode_begin(self, nepisodes, logs={}):
        self.SumQ = None
        self.NSteps = 0
        self.Actions = np.zeros((self.env.action_space.n,))

    def on_action_end(self, actions, logs={}):
        for a, qv in logs["qvectors"]:
            if self.SumQ is None:
                self.SumQ = qv
            else:
                self.SumQ += qv
            self.NSteps += 1
        for a, action in logs["actions"]:
            self.Actions[action] += 1
    
    def on_episode_end(self, episode, logs):
        avq = self.SumQ/self.NSteps if self.NSteps > 0 else 0.0
        rewards = [r for t, r in logs["episode_rewards"]]
        #shots = [t.State.Shots for t, r in logs["episode_rewards"]]
        action_frequencies = self.Actions/np.sum(self.Actions)
        print "Episode end: %d, rounds: %d, rewards: %s, average q: %s, actions: %s" % \
            (episode, logs["nrounds"], rewards, avq, self.Actions)

class QVectorLogger(Callback):
    
    def on_episode_begin(self, *__, **_):
        self.LogFile = open("/tmp/qvector.log", "w")
    
    def on_action_end(self, action_list, logs):
        o = logs["observations"][0][1]
        qv = logs["qvectors"][0][1]
        self.LogFile.write(" ".join(["%.3f" % (x,) for x in (list(o)+list(qv))]) + "\n")
        
    def on_episode_end(self, *_, **__):
        self.LogFile.close()

opts, args = getopt.getopt(sys.argv[1:], "k:r:h?g:")
opts = dict(opts)
kind = opts.get("-k", "diff")
run_log = opts.get("-r", "run_log.csv")
gamma = float(opts.get("-g", 0.8))

if "-h" in opts or "-?" in opts:
    print """Usage:
         python tanks_target.py [-k kind] [-r <run log CVS file>]
    """
    sys.exit(1)

env = GymEnv(MountainCarEnv(), tlimit=200)

print "Environment initialized:", env
agent = Agent(env, kind=kind, gamma=gamma)

controller = SynchronousMultiAgentController(env, [agent],                 
    rounds_between_train = 1000, episodes_between_train = 1)

taus = [1.0,0.1,0.01,0.001]
ntaus = len(taus)
t = 0

test_policy = BoltzmannQPolicy(0.0001)
#print "test policy:", test_policy
train_run_logger = RunLogger()
test_run_logger = RunLogger(run_log, loss_info_from=train_run_logger)

for _ in range(20000):
    
    controller.randomMoves([agent], 100)
    
    for i in range(2*ntaus):
        tau = taus[t%ntaus]
        policy = BoltzmannQPolicy(tau)
        print "-- Training with tau=%.4f..." % (tau,)
        controller.fit(max_episodes=20, callbacks=[train_run_logger], policy=policy)
        t += 1
    print "-- Testing..."
    controller.test(max_episodes=50, callbacks=[test_run_logger], policy=test_policy)
    controller.test(max_episodes=3, callbacks=[Visualizer(), EpisodeLogger()], policy=test_policy)
    #controller.test(max_episodes=20, callbacks=[], policy=test_policy)
    

    
