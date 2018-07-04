from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad
from policies import GreedyEpsPolicy
from callbacks import Callback, Visualizer
from MultiDQNAgent import MultiDQNAgent
from QNet import QNet
from tanks2_env import TankDuelEnv
from MultiAgentController import SynchronousMultiAgentController
from QBrain import QBrain

import numpy as np

np.set_printoptions(precision=4, suppress=True)

def create_model(inp_width, out_width):
    inp = Input((inp_width,))
    dense1 = Dense(55, activation="relu", name="dense1")(inp)
    dense2 = Dense(55, activation="relu", name="dense2")(dense1)
    dense3 = Dense(55, activation="relu", name="dense3")(dense2)
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
            print "Train session %d: loss=%f" % (nsessions, math.sqrt(logs["metrics"][0]))
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass

class EpisodeLogger(Callback):
    
    def on_episode_end(self, episode, logs):
        print "Episode end: %d, rounds: %d" % (episode, logs["nrounds"])

env = TankDuelEnv()
model = create_model(env.observation_space.shape[-1], env.actions_space.shape[-1])
brain = QBrain(model, soft_update=0.0001)
brain.compile(Adam(lr=1e-3), ["mse"])

tanks = [TankAgent(env, brain, test_policy=GreedyEpsPolicy(0.1), train_rounds=1000, steps_between_train=1001), 
        TankAgent(env, brain, test_policy=GreedyEpsPolicy(0.1), train_rounds=10, steps_between_train=1001)]
controller = SynchronousMultiAgentController(env, tanks)

for t in range(2000):
    for tank in tanks:
        tank.TrainPolicy.Epsilon = 0.1 if t % 3 == 0 else 0.5
    print "training..."
    controller.fit(max_episodes=5, callbacks=[TrainCallback(), EpisodeLogger()])
    print "testing..."
    controller.test(max_episodes=1, callbacks=[Visualizer()])
    

    
