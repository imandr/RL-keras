from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import QNet, DifferentialQNet
from RLKeras.multi import MultiDQNAgent, SynchronousMultiAgentController, QBrain
from RLKeras.callbacks import Callback, Visualizer
from RLKeras.policies import GreedyEpsPolicy


from lander_env_multi import LanderEnvMulti
import numpy as np
import math

np.set_printoptions(precision=4, suppress=True)

def create_model(inp_width, out_width):
    inp = Input((inp_width,))
    dense1 = Dense(25, activation="tanh", name="dense1")(inp)
    dense2 = Dense(25, activation="tanh", name="dense2")(dense1)
    dense3 = Dense(25, activation="tanh", name="dense3")(dense2)
    out = Dense(out_width, activation="linear", name="out_linear")(dense3)
    model = Model(inp, out)
    print("--- model summary ---")
    print(model.summary())
    return model
    
    
class LanderAgent(MultiDQNAgent):
    pass

class TrainCallback(Callback):
    
    def on_train_session_end(self, nsessions, logs):
        #if nsessions % 100 == 0:
        print "                                                    Train session %d: loss=%f" % (nsessions, math.sqrt(logs["metrics"]))
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass

class EpisodeLogger(Callback):
    
    def on_episode_end(self, episode, logs):
        print "Episode end: %d, rounds: %d, reward:%f" % (episode, logs["nrounds"],
            logs["episode_rewards"][0][1])
            
class QVectorLogger(Callback):
    
    def on_episode_begin(self, *__, **_):
        self.LogFile = open("/tmp/qvector.log", "w")
    
    def on_action_end(self, action_list, logs):
        o = logs["observations"][0][1]
        qv = logs["qvectors"][0][1]
        self.LogFile.write(" ".join(["%.3f" % (x,) for x in (list(o)+list(qv))]) + "\n")
        
    def on_episode_end(self, *_, **__):
        self.LogFile.close()

env = LanderEnvMulti()
model = create_model(env.observation_space.shape[-1], 4)
brain = QBrain(model, soft_update=0.01, gamma=0.99)
brain.compile(Adam(lr=1e-3), ["mse"])

lander = LanderAgent(env, brain,
        train_policy = GreedyEpsPolicy(0.5),      
        test_policy = GreedyEpsPolicy(0.0),
        train_sample_size = 2000, train_rounds = 1, trains_between_updates = 10
)

controller = SynchronousMultiAgentController(env, [lander],                 
    rounds_between_train = 500, episodes_between_train = 1)

epsilons = [0.01, 0.1, 0.2]
nepsilons = len(epsilons)
for t in range(2000):
    epsilon = epsilons[t%nepsilons]
    print "Epsilon:", epsilon
    policy = GreedyEpsPolicy(epsilon)
    controller.fit(max_episodes=5, callbacks=[TrainCallback(), EpisodeLogger()], policy=policy)
    print "Brain train_samples=",brain.trainSamples, "  age=",brain.age, "  memory size:", brain.recordSize()
    controller.test(max_episodes=1, callbacks=[Visualizer(), EpisodeLogger(), QVectorLogger()])
    

    
