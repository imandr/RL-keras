from ..callbacks import CallbackList
class MultiAgentController:
    
    def __init__(self, env, agents, callbacks = None,
            episodes_between_train = 1, rounds_between_train = 100
        ):
        self.Env = env
        self.Agents = agents
        self.Callbacks = callbacks
        
        self.EpisodesBetweenTrain = episodes_between_train
        self.RoundsBetweenTrain = rounds_between_train
        
        self.EpisodesToTrain = episodes_between_train
        self.RoundsToTrain = episodes_between_train
        
    
    def fit(self, max_episodes = None, max_rounds = None, max_steps = None, callbacks = None, policy=None):
        #print "fit"
        return self.run(self.Env, self.Agents, max_episodes, max_rounds, max_steps, 
            callbacks or self.Callbacks, True, policy)

    def test(self, max_episodes = None, max_rounds = None, max_steps = None, callbacks = None, policy=None):
        #print "test"
        return self.run(self.Env, self.Agents, max_episodes, max_rounds, max_steps, 
            callbacks or self.Callbacks, False, policy)
    
    def run(self, env, agents, max_episodes, max_steps, callbacks, training, policy):
        raise NotImplementedError

class SynchronousMultiAgentController(MultiAgentController):
    
    def run(self, env, agents, max_episodes, max_rounds, max_steps, callbacks, training, policy):
        
        callbacks = callbacks or []
        
        callbacks = CallbackList(callbacks)
        callbacks._set_env(env)
        
        self.Callbacks = callbacks
        
        assert max_episodes is not None or max_steps is not None
        
        nepisodes = 0
        nrounds = 0
        nsteps = 0
        
        while (max_episodes is None or nepisodes < max_episodes) and \
                    (max_steps is None or nsteps < max_steps):
                    
            for a in agents:
                a.episodeBegin()        # <-----
                a.Done = False
            
            env.reset(agents)        # <----
            active_agents = agents[:]
            
            episode_rewards = {id(agent):0.0 for agent in agents}
            
            callbacks.on_episode_begin(nepisodes, {
                'agents': agents
            })

            episode_round = 0
            episode_step = 0

            #print "episode begin: active_agents:", len(active_agents)
            #print agents[0].Done

            while active_agents:
                
                callbacks.on_round_begin(episode_round, {
                    'episode': nepisodes,
                    'episode_round': episode_round,
                    'agents': agents,
                    'active_agents': active_agents
                })

                observations_list = env.observe(active_agents)      # [(agent, obs, valids, done), ...]
                #print observations_list
                active_agents = []
                actions_list = []


                for agent, observation, valids, done, info in observations_list:
                    if done:
                        agent.Done = True
                        agent.final(observation, training)
                    else:
                        active_agents.append(agent)
                        actions_list.append((agent, agent.action(observation, valids, training, policy=policy)))

                feedback = []
                #print "active_agents:", len(active_agents)
                if active_agents:
                    step_infos = env.step(actions_list)
                
                    feedback = env.feedback(active_agents)
                    
                    for agent, reward, info in feedback:
                        episode_rewards[id(agent)] += reward
                
                    callbacks.on_action_end(actions_list, 
                        {
                            'observations': observations_list,
                            'actions': actions_list,
                            'feedback': feedback
                        })
                
                    metrics = None
                    if training:
                        actions = [action for agent, action in actions_list]
                        metrics = [(agent, agent.learn(actions[i], reward))
                                    for i, (agent, reward, _) in enumerate(feedback)]
                                
                step_logs = {
                    'episode': nepisodes,
                    'episode_step': episode_step,
                    'episode_round': episode_step,
                    'agents': agents,
                    'active_agents': active_agents,
                    'actions': actions_list,
                    'feedback': feedback,
                }
                callbacks.on_step_end(episode_step, step_logs)


                episode_round += 1
                episode_step += 1
                nsteps += 1
                nrounds += 1
                
                self.RoundsToTrain -= 1
                if training:    self.trainIfNeeded(agents)
                
            callbacks.on_episode_end(nepisodes, {
                'nrounds': nrounds,
                'agents': agents,
                'episode_rewards': [(agent, episode_rewards[id(agent)]) for agent in agents],
                'nsteps': nsteps
            })
            
            nepisodes += 1

            self.EpisodesToTrain -= 1
            if training:    self.trainIfNeeded(agents)
    
    def trainIfNeeded(self, agents):
        if self.RoundsToTrain <= 0 or self.EpisodesToTrain <= 0:
            for a in agents:
                a.trainBrain(self.Callbacks)
            self.RoundsToTrain = self.RoundsBetweenTrain
            self.EpisodesToTrain = self.EpisodesBetweenTrain

                 
class SequentialMultiAgentController(MultiAgentController):
    
    def run(self, env, agents, max_episodes, max_rounds, max_steps, callbacks, training):
        
        callbacks = callbacks or []
        
        callbacks = CallbackList(callbacks)
        callbacks._set_env(env)
        
        self.Callbacks = callbacks
        
        assert max_episodes is not None or max_steps is not None
        
        nepisodes = 0
        nrounds = 0
        nsteps = 0
        
        callbacks.on_run_begin(env, 
            {
                'max_episodes': max_episodes,
                'max_rounds': max_rounds,
                'max_steps': max_steps,
                'training': training,
                'agents':   agents
            }
        )
        while (max_episodes is None or nepisodes < max_episodes) and \
                    (max_steps is None or nsteps < max_steps):
                    
            for a in agents:
                a.episodeBegin()        # <-----
                a.Done = False
            
            env.reset(agents)        # <----
            active_agents = agents[:]

            episode_round = 0
            episode_step = 0
            
            callbacks.on_episode_begin(nepisodes, {
                'agents': agents
            })
            
                
            while active_agents:
                
                callbacks.on_round_begin(episode_round, {
                    'episode': nepisodes,
                    'episode_round': episode_round,
                    'agents': agents,
                    'active_agents': active_agents
                })

                new_active_agents = []
                
                for agent in active_agents:
                    observation, valids = env.observe(agent)
                    
                    action = agent.action(observation, valid_actions, training)
                    
                    env.step(action, agent)
                    
                    reward, done, info = env.feedback(agent)
                    
                    metrics = agent.learn(action, reward, done)

                    step_logs = {
                        'episode': nepisodes,
                        'episode_round': episode_round,
                        'episode_step': episode_step,
                        'agents': agents,
                        'active_agents': active_agents,
                        'agent': agent,
                        'actions': action,
                        'feedback': (reward, done, info),
                        'metrics': metrics
                    }
                    
                    callbacks.on_step_end(episode_step, step_logs)

                    if not done:
                        new_active_agents.append(agent)
                        
                    episode_step += 1
                    nsteps += 1
                
                callbacks.on_round_end(episode_round, {
                    'episode': nepisodes,
                    'episode_round': episode_round,
                    'episode_steps': episode_step,
                    'agents': agents,
                    'active_agents': active_agents
                })
                episode_round += 1
                nrounds += 1
            
            callbacks.on_episode_end(nepisodes, {
                'nrounds': episode_round,
                'agents': agents
            })
            
            nepisodes += 1
        callbacks.on_run_end(env, {
            'agents':   agents,
            'training': training,
            'nepisodes':    nepisodes,
            'nrounds':  nrounds,
            'nsteps':   nsteps
            }
        )
