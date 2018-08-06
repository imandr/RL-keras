import gym, math
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad

from RLKeras import GreedyEpsPolicy, QNet, ReplayMemory
from RLKeras.single import DQNAgent
from RLKeras.callbacks import Visualizer, TestLogger, Callback, TrainEpisodeLogger
from env import LunarLander

def create_model(inp_width, out_width):
    inp = Input((inp_width,))
    dense1 = Dense(25, activation="relu", name="dense1")(inp)
    dense2 = Dense(25, activation="relu", name="dense2")(dense1)
    dense3 = Dense(25, activation="relu", name="dense3")(dense2)
    out = Dense(out_width, activation="linear", name="out_linear")(dense3)
    model = Model(inp, out)
    print("--- model summary ---")
    print(model.summary())
    return model


class TrainCallback(Callback):
    
    def on_train_session_end(self, nsessions, logs):
        #if nsessions % 100 == 0:
            print "Train session %d: loss=%f" % (nsessions, math.sqrt(logs["metrics"]))
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass
        
class EpisodeLogger(Callback):
    
    def on_episode_end(self, episode, logs):
        print "Episode end: %d, steps: %d, episdode reward:%f" % (episode, logs["nb_episode_steps"], logs["episode_reward"])

class Lander(DQNAgent):
    
    ValidActions = [0,1,2,3]
    
    def __init__(self, env, qnet):
        self.TrainPolicy = GreedyEpsPolicy(0.5)      
        self.TestPolicy = GreedyEpsPolicy(0.0)
        self.Memory = ReplayMemory(100000)
        DQNAgent.__init__(self, env, qnet, self.Memory, gamma=0.99,
                train_policy=self.TrainPolicy, test_policy=self.TestPolicy,
                steps_between_train = 500, episodes_between_train = 1, 
                train_sample_size = 20, train_rounds = 100,
                trains_between_updates = 1
        )
    
    def updateState(self, observation):
        return self.ValidActions


#env = gym.make("LunarLander-v2")
env = LunarLander()
model = create_model(env.observation_space.shape[-1], 4)
qnet = QNet(model, 0.01)
qnet.compile(Adam(lr=1e-3), ["mse"])

lander = Lander(env, qnet)

for t in range(2000):
    lander.TrainPolicy.Epsilon = 0.1 if t % 2 == 0 else 0.5
    print "epsilon:", lander.TrainPolicy.Epsilon 
    lander.fit(max_episodes=5, callbacks=[TrainCallback(), EpisodeLogger()])
    #lander.TrainPolicy.Epsilon = max(lander.TrainPolicy.Epsilon*0.95, 0.2)
    print "QNet train_samples=",qnet.TrainSamples, "  memory age=",lander.age
    lander.test(max_episodes=1, callbacks=[TestLogger(), Visualizer(), TrainCallback()])
    



