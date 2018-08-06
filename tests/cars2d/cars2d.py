from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import QNet, QBrain
from RLKeras.multi import MultiDQNAgent, SynchronousMultiAgentController
from RLKeras.callbacks import Callback, Visualizer
from RLKeras.policies import BoltzmanPolicy

from cars2d_env import CarsRadEnv
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def create_model(inp_width, out_width):
    inp = Input((inp_width,))
    dense1 = Dense(35, activation="tanh", name="dense1")(inp)
    dense2 = Dense(35, activation="softplus", name="dense2")(dense1)
    dense3 = Dense(35, activation="softplus", name="dense3")(dense2)
    out = Dense(out_width, activation="linear", name="out_linear")(dense3)
    model = Model(inp, out)
    print("--- model summary ---")
    print(model.summary())
    return model
    
class CarAgent(MultiDQNAgent):
    pass

class TrainCallback(Callback):
    
    def on_train_session_end(self, nsessions, logs):
        print "Train session %d: loss=%f" % (nsessions, math.sqrt(logs["metrics"]))
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass

class EpisodeLogger(Callback):
    
    def on_episode_end(self, episode, logs):
        rewards = [r for t, r in logs["episode_rewards"]]
        print "Episode end: %d, rounds: %d, rewards: %s" % (episode, logs["nrounds"], rewards)


opts, args = getopt.getopt(sys.argv[1:], "k:r:h?")
opts = dict(opts)
kind = opts.get("-k", "diff")
run_log = opts.get("-r", "run_log.csv")
if "-h" in opts or "-?" in opts:
    print """Usage:
         python cars2d.py [-k kind] [-r <run log CVS file>]
    """
    sys.exit(1)


env = CarsRadEnv()
model = create_model(env.observation_space.shape[-1], env.actions_space.shape[-1])
brain = QBrain(model, kind="diff", gamma=0.99)      #, soft_update=0.01)
brain.compile(Adam(lr=1e-3), ["mse"])

cars = [CarAgent(env, brain, train_sample_size=1000) for _ in range(3)]

#cars = []
#for _ in range(3):
#    model = create_model(env.observation_space.shape[-1], env.actions_space.shape[-1])
#    brain = QBrain(model, soft_update=0.01)
#    brain.compile(Adam(lr=1e-3), ["mse"])
#    cars.append(CarAgent(env, brain))
    



controller = SynchronousMultiAgentController(env, cars,
    rounds_between_train = 10000, episodes_between_train = 1
    )

#controller.randomMoves(env, cars, 30000, callbacks=[TrainCallback()])

taus = [2.0, 1.0, 0.1, 0.01]
ntaus = len(taus)
t = 0

test_policy = BoltzmannQPolicy(0.005)

train_run_logger = RunLogger()
test_run_logger = RunLogger(run_log, loss_info_from=train_run_logger)

for _ in range(20000):
    for i in range(2*ntaus):
        tau = taus[t%ntaus]
        policy = BoltzmannQPolicy(tau)
        print "Tau=%f, training..." % (tau,)
        controller.fit(max_episodes=10, callbacks=[train_run_logger], policy=policy)
        t += 1
    print "-- Testing..."
    controller.test(max_episodes=50, callbacks=[test_run_logger], policy=test_policy)
    controller.test(max_episodes=5, callbacks=[Visualizer(), EpisodeLogger()], policy=test_policy)
    

    
