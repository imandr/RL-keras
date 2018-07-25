from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import QNet, QBrain
from RLKeras.multi import MultiDQNAgent, SynchronousMultiAgentController
from RLKeras.callbacks import Callback, Visualizer
from RLKeras.policies import GreedyEpsPolicy, BoltzmannQPolicy

from tanks_target_env import TankTargetEnv
import math
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def create_model(inp_width, out_width):
    inp = Input((inp_width,))
    dense1 = Dense(25, activation="tanh", name="dense1")(inp)
    dense2 = Dense(50, activation="softplus", name="dense2")(dense1)
    dense3 = Dense(20, activation="softplus", name="dense3")(dense2)
    out = Dense(out_width, activation="linear", name="out_linear")(dense3)
    model = Model(inp, out)
    print("--- model summary ---")
    print(model.summary())
    return model
    
class TankAgent(MultiDQNAgent):
    pass

class TrainRunLog(Callback):
    
    def on_train_session_end(self, nsessions, logs):
        pass
        
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass
        
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
        print "Train episodes/rounds/steps:", \
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
        self.Actions = np.zeros((self.env.action_space.shape[-1],))

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
        action_frequencies = self.Actions/np.sum(self.Actions)
        print "Episode end: %d, rounds: %d, rewards: %s, average q: %s, actions: %s" % \
            (episode, logs["nrounds"], rewards, avq, self.Actions)

env = TankTargetEnv()
model = create_model(env.observation_space.shape[-1], env.action_space.shape[-1])
brain = QBrain(model, typ="diff", v_selectivity=False, qnet_hard_update=2000, gamma=0.99)
brain.compile(Adam(lr=1e-3), ["mse"])

tanks = [TankAgent(env, brain, train_sample_size=1000) for _ in range(1)]
#, 
#        TankAgent(env, brain, test_policy=GreedyEpsPolicy(0.0), train_sample_size = 20, train_rounds = 50)]
controller = SynchronousMultiAgentController(env, tanks,
    rounds_between_train = 10000, episodes_between_train = 1
    )

taus = [0.01, 0.1, 1.0, 2.0]
ntaus = len(taus)
t = 0

test_policy = BoltzmannQPolicy(0.005)

train_run_logger = RunLogger()
test_run_logger = RunLogger("run_log.csv", loss_info_from=train_run_logger)

for _ in range(20000):
    for i in range(2*ntaus):
        t += 1
        tau = taus[t%ntaus]
        policy = BoltzmannQPolicy(tau)
        print "Tau=%f, training..." % (tau,)
        controller.fit(max_episodes=10, callbacks=[train_run_logger], policy=policy)
    print "-- Testing..."
    controller.test(max_episodes=50, callbacks=[test_run_logger], policy=test_policy)
    controller.test(max_episodes=3, callbacks=[Visualizer(), EpisodeLogger()], policy=test_policy)
    

    
