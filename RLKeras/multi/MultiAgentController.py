from ..callbacks import CallbackList
import random

class MultiAgentController:
    
    def __init__(self, env, agents,
            episodes_between_train = 1, rounds_between_train = 100
        ):
        self.Env = env
        self.Agents = agents
        
        self.EpisodesBetweenTrain = episodes_between_train
        self.RoundsBetweenTrain = rounds_between_train
        
        self.EpisodesToTrain = episodes_between_train
        self.RoundsToTrain = episodes_between_train
        
        self.TotalTrainSteps = 0
        self.TotalTrainRounds = 0
        self.TotalTrainEpisodes = 0
        
    
    def fit(self, max_episodes = None, max_rounds = None, max_steps = None, callbacks = None, policy=None):
        #print "fit"
        return self.run(self.Env, self.Agents, max_episodes, max_rounds, max_steps, 
            callbacks, True, policy)
            
    def random_fit(self, max_rounds = None, callbacks = None):
        #print "fit"
        return self.randomMoves(self.Agents, max_rounds, callbacks=callbacks) 
    

    def test(self, max_episodes = None, max_rounds = None, max_steps = None, callbacks = None, policy=None):
        #print "test"
        return self.run(self.Env, self.Agents, max_episodes, max_rounds, max_steps, 
            callbacks, False, policy)
    
    def run(self, env, agents, max_episodes, max_steps, callbacks, training, policy):
        raise NotImplementedError

class SynchronousMultiAgentController(MultiAgentController):
    
    def randomMoves(self, agents, max_rounds, callbacks = None):

        callbacks = callbacks or []
        
        callbacks = CallbackList(callbacks)
        callbacks._set_env(self.Env)

        for sample in xrange(max_rounds):
            moves = self.Env.randomMoves(agents)
            for agent, obs0, action, obs1, reward, done, valids, info in moves:
                #print "randomMoves: memorizing:", obs0, action, reward, obs1, done, valids
                agent.memorize((obs0, action, reward, obs1, done, valids), reward)
            self.RoundsToTrain -= 1
            self.trainIfNeeded(agents, callbacks)

            #self.Brain.memorize((self.Observation0, self.Action0, self.Reward0, 
            #    self.Observation1, False, self.Valids1), self.Reward0)    

    
    def run(self, env, agents, max_episodes, max_rounds, max_steps, callbacks, training, policy):
        
        callbacks = callbacks or []
        
        callbacks = CallbackList(callbacks)
        callbacks._set_env(env)
                
        assert max_episodes is not None or max_steps is not None
        
        nepisodes = 0
        nrounds = 0
        nsteps = 0
        
        callbacks.on_run_begin(None,
            dict(
                training=training,
                agents=agents,
                max_episodes=max_episodes,
                max_rounds=max_rounds,
                max_steps=max_steps,
                policy=policy
                )
        )
        
        run_rewards = {id(a):0.0 for a in agents}
        
        while (max_episodes is None or nepisodes < max_episodes) and (max_steps is None or nsteps < max_steps):
                    
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

            while active_agents:    # and not env.over():
                
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
                        actions_list.append((agent, agent.action(observation, valids, training, policy)))

                feedback = []
                #print "active_agents:", len(active_agents)
                if active_agents:
                    step_infos = env.step(actions_list)
                
                    feedback = env.feedback(active_agents)
                    
                    if training:
                        self.TotalTrainSteps += len(actions_list)

                    for agent, reward, info in feedback:
                        episode_rewards[id(agent)] += reward
                
                    callbacks.on_action_end(actions_list, 
                        {
                            'qvectors': [(agent, agent.QVector) for agent in active_agents],
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
                    'total_train_steps': self.TotalTrainSteps,
                    'total_train_rounds': self.TotalTrainRounds,
                    'total_train_episodes': self.TotalTrainEpisodes,
                }
                callbacks.on_step_end(episode_step, step_logs)


                if training:
                    self.TotalTrainRounds += 1

                episode_round += 1
                episode_step += 1
                nsteps += 1
                nrounds += 1
                
                self.RoundsToTrain -= 1
                if training:    self.trainIfNeeded(agents, callbacks)
                
            callbacks.on_episode_end(nepisodes, {
                'nrounds': nrounds,
                'agents': agents,
                'episode_rewards': [(agent, episode_rewards[id(agent)]) for agent in agents],
                'nsteps': nsteps,
                'total_train_steps': self.TotalTrainSteps,
                'total_train_rounds': self.TotalTrainRounds,
                'total_train_episodes': self.TotalTrainEpisodes,
            })
            
            for aid, r in episode_rewards.items():
                run_rewards[aid] += r
                
            
            nepisodes += 1
            if training:
                self.TotalTrainEpisodes += 1

            self.EpisodesToTrain -= 1
            if training:    self.trainIfNeeded(agents, callbacks)

        callbacks.on_run_end(None,            
            dict(
                nepisodes=nepisodes,
                nrounds=nrounds,
                nsteps=nsteps,
                run_rewards = [(agent, run_rewards[id(agent)]) for agent in agents],
                total_train_steps = self.TotalTrainSteps,
                total_train_rounds = self.TotalTrainRounds,
                total_train_episodes = self.TotalTrainEpisodes
                )
        )

    
    def trainIfNeeded(self, agents, callbacks):
        if self.RoundsToTrain <= 0 or self.EpisodesToTrain <= 0:
            #print "train all brains...", self.RoundsToTrain, self.EpisodesToTrain
            for a in agents:
                a.trainBrain(callbacks)
            self.RoundsToTrain = self.RoundsBetweenTrain
            self.EpisodesToTrain = self.EpisodesBetweenTrain

                 
class SequentialMultiAgentController(MultiAgentController):
    
    def run(self, env, agents, max_episodes, max_rounds, max_steps, callbacks, training):
        
        callbacks = callbacks or []
        
        callbacks = CallbackList(callbacks)
        callbacks._set_env(env)
        
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
