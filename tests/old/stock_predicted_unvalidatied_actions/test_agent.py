import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam, Adagrad
from policies import GreedyEpsPolicy

from DQNAgent import DQNAgent, QNet
from callbacks import Visualizer, TestLogger, Callback

from test_env import PredictedEnv, Stock

def create_model(inp_width, out_width):
    inp = Input((inp_width,))
    dense1 = Dense(50, activation="tanh", name="dense1")(inp)
    dense2 = Dense(20, activation="tanh", name="dense2")(dense1)
    dense3 = Dense(out_width, activation="linear", name="out_linear")(dense2)
    model = Model(inp, dense3)
    print("--- model summary ---")
    print(model.summary())
    return model
    
class ActionCallback(Callback):
    
    def format_array(self, fmt, array):
        return "[%s]" % (" ".join([fmt % (x,) for x in array]),)
    
    def on_step_end(self, episode_step, logs):
        sagent = "o:%s v:%11s a:%1d q:%s reward:%7.4f" % (
            self.format_array("%7.4f", logs["observation"]),
            self.format_array("%1d", logs["valid_actions"]),
            logs["action"],
            self.format_array("%7.4f", logs["qvector"]),
            logs["reward"])
        senv = "ratio:%7.4f growth:%8.4f value:%7.1f stock:%7.1f" % (self.env.ratio, self.env.StockGrowth, self.env.value,
                    self.env.Stock.P)
        print sagent, senv
    

class PredictedAgent(DQNAgent):
    
    def __init__(self, env, qnet):
        self.TrainPolicy = GreedyEpsPolicy(0.5)      
        self.TestPolicy = GreedyEpsPolicy(0.0)
        DQNAgent.__init__(self, env, qnet, gamma=0.9,
                train_policy=self.TrainPolicy, test_policy=self.TestPolicy,
                steps_between_train = 1000, episodes_between_train = 1, 
                train_sample_size = 50, train_rounds = 40,
                trains_between_updates = 1
        )

    def updateState(self, observation):
        return self.Env.validActions()

# Get the environment and extract the number of actions.
stock = Stock(100.0, 0.2, 0.0, 252)
env = PredictedEnv(stock, 1000.0, 0.0, 252)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.

model = create_model(env.observation_space.shape[-1], env.action_space.n)

qnet = QNet(model)
qnet.compile(Adagrad(), ["mse"])

agent = PredictedAgent(env, qnet)
for _ in range(2000):
    agent.fit(max_steps=50000)
    agent.TrainPolicy.Epsilon = max(agent.TrainPolicy.Epsilon*0.8, 0.1)
    print "epsilon:", agent.TrainPolicy.Epsilon 
    agent.test(max_episodes=1, callbacks=[TestLogger(), ActionCallback()])
    
    
agent.test(max_episodes=10, callbacks=[Visualizer(), TestLogger(), ActionCallback()])
