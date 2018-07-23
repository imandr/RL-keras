from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import QNet
from RLKeras.multi import MultiDQNAgent, SynchronousMultiAgentController, QBrain
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

class TrainCallback(Callback):
    
    def on_train_session_end(self, nsessions, logs):
        print "Train session %d: loss=%f" % (nsessions, math.sqrt(logs["metrics"]))
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass
        
class RunLogger(Callback):
    
    def __init__(self, csv_out = None):
        if isinstance(csv_out, str):
            csv_out = open(csv_out, "w")
        if csv_out is not None:
            csv_out.write("episodes,rounds,steps,reward_per_episode\n")
        self.CSVOut = csv_out
    
    def on_run_end(self, param, logs={}):
        rewards = logs["run_rewards"]
        nagents = len(rewards)
        nepisodes = logs["nepisodes"]
        avg_reward = float(sum([r for t, r in rewards]))/nepisodes/nagents
        print "Train episodes/rounds/steps:", \
            logs["total_train_episodes"], \
            logs["total_train_rounds"], \
            logs["total_train_steps"], \
        "Session reward per episode = ", avg_reward
        if self.CSVOut is not None:
            self.CSVOut.write("%d,%d,%d,%f\n" % ( 
                    logs["total_train_episodes"],
                    logs["total_train_rounds"],
                    logs["total_train_steps"],
                    avg_reward
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
brain = QBrain(model, typ="diff", v_selectivity=True, soft_update=0.01, gamma=0.99)
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

test_run_logger = RunLogger("run_log.csv")

for _ in range(20000):
    for i in range(2*ntaus):
        t += 1
        tau = taus[t%ntaus]
        policy = BoltzmannQPolicy(tau)
        print "Tau=%f, training..." % (tau,)
        controller.fit(max_episodes=10, callbacks=[RunLogger()], policy=policy)
    print "-- Testing..."
    controller.test(max_episodes=50, callbacks=[test_run_logger], policy=test_policy)
    controller.test(max_episodes=3, callbacks=[Visualizer(), EpisodeLogger()], policy=test_policy)
    

    
