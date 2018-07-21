from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import QNet, GymEnv
from RLKeras.multi import MultiDQNAgent, SynchronousMultiAgentController, QBrain
from RLKeras.callbacks import Callback, Visualizer
from RLKeras.policies import GreedyEpsPolicy

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

env = GymEnv("CartPole-v1")

class CartPoleAgent(MultiDQNAgent):
    pass

class TrainCallback(Callback):
    
    def on_train_session_end(self, nsessions, logs):
        print "Train session %d: loss=%f" % (nsessions, math.sqrt(logs["metrics"]))
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass

class EpisodeLogger(Callback):
    
    def on_episode_begin(self, nepisodes, logs={}):
        self.SumQ = None
        self.NSteps = 0

    def on_action_end(self, actions, logs={}):
        for a, qv in logs["qvectors"]:
            if self.SumQ is None:
                self.SumQ = qv
            else:
                self.SumQ += qv
            self.NSteps += 1
    
    def on_episode_end(self, episode, logs):
        avq = self.SumQ/self.NSteps if self.NSteps > 0 else 0.0
        rewards = [r for t, r in logs["episode_rewards"]]
        print "Episode end: %d, rounds: %d, rewards: %s, average q: %s" % (episode, logs["nrounds"], rewards, avq)

model = create_model(env.observation_space.shape[-1], env.action_space.n)
brain = QBrain(model, soft_update=0.01, gamma=0.9)
brain.compile(Adam(lr=1e-3), ["mse"])

#cars = [CarAgent(env, brain) for _ in range(3)]

agents = [CartPoleAgent(env, brain)]

controller = SynchronousMultiAgentController(env, agents,
    rounds_between_train = 1000, episodes_between_train = 10
    )

#controller.randomMoves(env, cars, 30000, callbacks=[TrainCallback()])

epsilons = [0.01, 0.1, 0.5]
nepsilons = len(epsilons)

while True:
    for _ in range(2):
        for epsilon in epsilons:
            print "Epsilon:", epsilon
            policy = GreedyEpsPolicy(epsilon)
            print "training..."
            controller.fit(max_episodes=100, callbacks=[TrainCallback(), EpisodeLogger()], policy=policy)
    print "testing..."
    controller.test(max_episodes=3, callbacks=[Visualizer(), EpisodeLogger()])

    