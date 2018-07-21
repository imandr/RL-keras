from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import QNet
from RLKeras.multi import MultiDQNAgent, SynchronousMultiAgentController, QBrain
from RLKeras.callbacks import Callback, Visualizer
from RLKeras.policies import GreedyEpsPolicy

from cars2d_env import CarsRadEnv
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

env = CarsRadEnv()
model = create_model(env.observation_space.shape[-1], env.actions_space.shape[-1])
brain = QBrain(model, soft_update=0.01)
brain.compile(Adam(lr=1e-3), ["mse"])

#cars = [CarAgent(env, brain) for _ in range(3)]

cars = []
for _ in range(3):
    model = create_model(env.observation_space.shape[-1], env.actions_space.shape[-1])
    brain = QBrain(model, soft_update=0.01)
    brain.compile(Adam(lr=1e-3), ["mse"])
    cars.append(CarAgent(env, brain))
    



controller = SynchronousMultiAgentController(env, cars,
    rounds_between_train = 1000, episodes_between_train = 1
    )

#controller.randomMoves(env, cars, 30000, callbacks=[TrainCallback()])

epsilons = [0.01, 0.1, 0.5]
nepsilons = len(epsilons)

t = 0
#for t in range(2000):
while True:
    t += 1
    epsilon = epsilons[t%nepsilons]
    print "Epsilon:", epsilon
    policy = GreedyEpsPolicy(epsilon)
    print "training..."
    controller.fit(max_episodes=20, callbacks=[TrainCallback(), EpisodeLogger()], policy=policy)
    print "testing..."
    controller.test(max_episodes=1, callbacks=[Visualizer(), EpisodeLogger()])
    

    
