from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import QNet
from RLKeras.multi import MultiDQNAgent, SynchronousMultiAgentController, QBrain
from RLKeras.callbacks import Callback, Visualizer
from RLKeras.policies import GreedyEpsPolicy

from tanks2_env import TankDuelEnv
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def create_model(inp_width, out_width):
    inp = Input((inp_width,))
    dense1 = Dense(35, activation="relu", name="dense1")(inp)
    dense2 = Dense(35, activation="relu", name="dense2")(dense1)
    dense3 = Dense(35, activation="relu", name="dense3")(dense2)
    out = Dense(out_width, activation="linear", name="out_linear")(dense3)
    model = Model(inp, out)
    print("--- model summary ---")
    print(model.summary())
    return model
    
class TankAgent(MultiDQNAgent):
    pass

class TrainCallback(Callback):
    
    def on_train_session_end(self, nsessions, logs):
        if nsessions % 100 == 0:
            print "Train session %d: loss=%f" % (nsessions, math.sqrt(logs["metrics"]))
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass

class EpisodeLogger(Callback):
    
    def on_episode_end(self, episode, logs):
        rewards = [r for t, r in logs["episode_rewards"]]
        print "Episode end: %d, rounds: %d, rewards: %s" % (episode, logs["nrounds"], rewards)

env = TankDuelEnv()
model = create_model(env.observation_space.shape[-1], env.actions_space.shape[-1])
brain = QBrain(model, soft_update=0.001)
brain.compile(Adam(lr=1e-3), ["mse"])

tanks = [TankAgent(env, brain, test_policy=GreedyEpsPolicy(0.1), train_sample_size = 20, train_rounds = 50), 
        TankAgent(env, brain, test_policy=GreedyEpsPolicy(0.1), train_sample_size = 20, train_rounds = 50)]
controller = SynchronousMultiAgentController(env, tanks,
    rounds_between_train = 10000, episodes_between_train = 1
    )

for t in range(2000):
    epsilon = 0.1 if t % 2 == 0 else 0.5
    print "Epsilon:", epsilon
    policy = GreedyEpsPolicy(epsilon)
    print "training..."
    controller.fit(max_episodes=5, callbacks=[TrainCallback(), EpisodeLogger()], policy=policy)
    print "testing..."
    controller.test(max_episodes=1, callbacks=[Visualizer(), EpisodeLogger()])
    

    
