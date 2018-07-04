from policies import GreedyEpsPolicy
from callbacks import CallbackList

class Agent(object):
    
    def __init__(self, env, train_policy = None, test_policy = None):
        self.TrainPolicy = train_policy
        self.TestPolicy = test_policy
        self.Env = env
        self.Training = False
        
    def reset_states(self):
        pass
    
    def episodeBegin(self):
        pass
    
    def episodeEnd(self):
        pass
        
    def action(self, observation, policy):
        raise NotImplementedError
        
    def learn(self, reward, new_observation, final):
        raise NotImplementedError

    def final(self, observation):
        raise NotImplementedError
        
    @property
    def done(self):
        raise NotImplementedError
    
    def run(self, max_episodes, max_steps, callbacks, training):
        
        self.Training = training
        
        callbacks = callbacks or []
        
        callbacks = CallbackList(callbacks)
        callbacks._set_env(self.Env)
        
        self.Callbacks = callbacks
        
        policy = self.TrainPolicy if training else self.TestPolicy
        
        assert max_episodes is not None or max_steps is not None
        
        nsteps = 0
        nepisodes = 0
        callbacks.on_train_begin()
        while (max_episodes is None or nepisodes < max_episodes) and \
                    (max_steps is None or nsteps < max_steps):
            self.reset_states()
            self.episodeBegin()
            callbacks.on_episode_begin(nepisodes)
            self.Env.reset([self])
            episode_reward = 0.0
            episode_step = 0
            done = False
            while not done:
                observation, valid_actions = self.Env.observe([self])[0][1:2]
                action = self.action(observation, valid_actions, training)
                info = self.Env.step([(self, action)])[0][1]
                _, reward, done, info = self.Env.feedback([self])
                new_observation, _ = self.Env.observe([self])[0][1:2]
                callbacks.on_action_end(action, 
                    {
                        "valid_actions":valid_actions,
                        'observation': observation,
                        'new_observation': new_observation,
                        'reward': reward,
                        "done":done,
                        "info":info,
                    })
                nsteps += 1
                episode_reward += reward
                if training:
                    metrics, metrics_names = self.learn(action, reward, done)
                else:
                    metrics, metrics_names = None, None
                    
                step_logs = {
                    'done': done,
                    'observation': observation,
                    'valid_actions':valid_actions,
                    'qvector': self.LastQVector,
                    'action': action,
                    'new_observation': new_observation,
                    'reward': reward,
                    'episode': nepisodes,
                    'info': info,
                    'metrics': metrics,
                    'metrics_names': metrics_names,
                    'episode_step': episode_step
                }
                episode_step += 1
                callbacks.on_step_end(episode_step, step_logs)
                observation = new_observation
                if done:
                    self.final(observation)        # let the agent see the final state
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_episode_steps': episode_step,
                'nb_steps': nsteps,
                'episode': nepisodes
            }
            callbacks.on_episode_end(nepisodes, episode_logs)
            nepisodes += 1
            self.episodeEnd()
        callbacks.on_train_end(logs={'did_abort': False})
            
    def fit(self, max_episodes = None, max_steps = None, callbacks = None):
        #print "fit"
        return self.run(max_episodes, max_steps, callbacks, True)

    def test(self, max_episodes = None, max_steps = None, callbacks = None):
        #print "test"
        return self.run(max_episodes, max_steps, callbacks, False)
        
    def updateState(self, observation):
        # update any internal state and return list of valid actions
        raise NotImplementedError
