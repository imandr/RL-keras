import gym, math
from callbacks import Visualizer, TestLogger, Callback, TrainEpisodeLogger
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad
from policies import GreedyEpsPolicy
from DQNAgent import DQNAgent, QNet
from secretary_env import SecretarySelectionEnv
import numpy as np

def create_model(inp_width, out_width):
    inp = Input((inp_width,))
    dense1 = Dense(55, activation="relu", name="dense1")(inp)
    dense2 = Dense(35, activation="relu", name="dense2")(dense1)
    dense3 = Dense(15, activation="relu", name="dense3")(dense2)
    out = Dense(out_width, activation="linear", name="out_linear")(dense3)
    model = Model(inp, out)
    print("--- model summary ---")
    print(model.summary())
    return model


class TrainCallback(Callback):
    
    def on_train_session_end(self, nsessions, logs):
        if nsessions % 100 == 0:
            print "Train session %d: loss=%f" % (nsessions, math.sqrt(logs["metrics"][0]))
        
    def on_qnet_update(self, nupdates, logs):
        #print "QNet updated"
        pass
        
class EpisodeLogger(Callback):
    
    def on_episode_end(self, episode, logs):
        print "Episode end: %d, steps: %d, episdode reward:%f" % (episode, logs["nb_episode_steps"], logs["episode_reward"])

class SecretarySelector(DQNAgent):
    
    ValidActions = [0,1]
    
    def __init__(self, env, qnet):
        self.TrainPolicy = GreedyEpsPolicy(0.5)      
        self.TestPolicy = GreedyEpsPolicy(0.0)
        DQNAgent.__init__(self, env, qnet, gamma=0.99,
                train_policy=self.TrainPolicy, test_policy=self.TestPolicy,
                steps_between_train = 500, episodes_between_train = 1, 
                train_sample_size = 20, train_rounds = 5
        )
        

    def updateState(self, observation):
        return self.ValidActions
        
class HireLogger(Callback):
    
    def __init__(self):
        self.F = open("record.csv", "w")
        self.F.write("episode,reward,steps,ratio,smooted_reward,smooted_steps,smooted_ratio\n")
        
        self.Smoothed = None
        self.Smooth = 0.01
        self.N = 0
    
    def on_step_end(self, episode_step, logs):
        #print logs["done"]
        if logs.get('done'):
            info = logs["info"]
            if self.Smoothed is None:
                self.Smoothed = np.array([logs["reward"], logs["episode_step"],info["ratio"]])
            else:
                x = np.array([logs["reward"], logs["episode_step"],info["ratio"]])
                self.Smoothed += self.Smooth*(x-self.Smoothed)
            #self.F.write("%d,%f,%d,%f,%f,%f,%f\n" % 
            #    (logs["episode"], logs["reward"], logs["episode_step"],info["ratio"],
            #    self.Smoothed[0], self.Smoothed[1], self.Smoothed[2]))
            #self.F.flush()            
            
            self.N += 1
            if self.N % 100 == 0:
                print self.Smoothed[0],self.Smoothed[1],self.Smoothed[2] 

NCandidates = 20


#env = gym.make("LunarLander-v2")
env = SecretarySelectionEnv(NCandidates)
model = create_model(env.observation_space.shape[-1], 2)
qnet = QNet(model, 0.01)
qnet.compile(Adam(lr=1e-3), ["mse"])

agent = SecretarySelector(env, qnet)

hire_logger = HireLogger()

for t in range(2000):
    agent.TrainPolicy.Epsilon = 0.1 if t % 2 == 0 else 0.5
    #print "epsilon:", agent.TrainPolicy.Epsilon 
    agent.fit(max_episodes=5, callbacks=[])     #TrainCallback(), EpisodeLogger()])
    #lander.TrainPolicy.Epsilon = max(lander.TrainPolicy.Epsilon*0.95, 0.2)
    agent.test(max_episodes=10, callbacks=[hire_logger])



