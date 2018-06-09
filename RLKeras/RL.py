import random, time, math
from mynnet import Linear, Model, Tanh, Sigmoid, InputLayer, AdaDeltaApplier, L2Regression, LSTM
import numpy as np

class ReplayMemory:
    
    M = 1
    
    def __init__(self, size):
        self.MaxSize = size
        self.HighWater = int(size*1.3)
        self.Memory = []
        self.ShortTermMemory = []
        
    def add(self, tup):   
        self.ShortTermMemory.append(tup)
            
    def add_to_long(self, tups):
        self.Memory.extend(tups)
        if len(self.Memory) > self.HighWater:
            self.Memory = random.sample(self.Memory, self.MaxSize)
        
    def sample(self, size):
        shorts = self.ShortTermMemory[:size]
        n_short = len(shorts)
        self.ShortTermMemory = self.ShortTermMemory[n_short:]
        n_long = min(len(self.Memory), size - n_short)
        longs = random.sample(self.Memory, n_long) if n_long > 0 else []
        self.add_to_long(shorts)
        return shorts + longs
        
    def size(self):
        return len(self.Memory)+len(self.ShortTermMemory)

    def sizes(self):
        return len(self.Memory), len(self.ShortTermMemory)
        
class RollOverMemory(object):
    def __init__(self, size, short_term_size = 50):
        self.MaxSize = size
        self.MaxShortTermSize = short_term_size
        self.Memory = []
        self.ShortTermMemory = []
        self.I = 0
    
    def add(self, tup):   
        self.ShortTermMemory.append(tup)
        tups = []
        if len(self.ShortTermMemory) > self.MaxShortTermSize:
            n = len(self.ShortTermMemory) - self.MaxShortTermSize
            tups = self.ShortTermMemory[:n]
            self.ShortTermMemory = self.ShortTermMemory[n:]
            self.add_to_long(tups)
            
    def add_to_long(self, tups):
        if self.MaxSize > 0:
            for tup in tups:
                if len(self.Memory) >= self.MaxSize:
                    self.Memory[self.I] = tup
                    self.I = (self.I + 1) % self.MaxSize
                else:
                    self.Memory.append(tup)
        
    def sample(self, size):
        shorts = self.ShortTermMemory[:size]
        n_short = len(shorts)
        self.ShortTermMemory = self.ShortTermMemory[n_short:]
        n_long = min(len(self.Memory), size - n_short)
        longs = random.sample(self.Memory, n_long) if n_long > 0 else []
        self.add_to_long(shorts)
        return shorts + longs
        
    def size(self):
        return len(self.Memory)+len(self.ShortTermMemory)

    def sizes(self):
        return len(self.Memory), len(self.ShortTermMemory)
        
class Environment(object):
    
    StateVectorSize = None
    NActions = None
    
    def reset(self, *params, **args):
        pass

    def initialState(self):
        raise NotImplementedError

class State(object):
    
    def __init__(self, env):
        self.Env = env
        self.Vector = None
        self.Actions = None
        self.resetQ()
        
    def resetQ(self):
        self.QV = None
        self.QSorted = None
        
    @property
    def actions(self):
        if self.Actions is None:    self.Actions = self.possibleActions()
        return self.Actions

    def possibleActions(self):
        raise NotImplementedError
        
    """
        def qsorted(self, qnet):
        if self.QSorted is None:
            q = self.qvector(qnet)
            self.QSorted = sorted([(q[a], a) for a in self.actions])[::-1]
        return self.QSorted
        
    def qvector(self, qnet):
        if self.QV is None:
            self.QV = qnet.compute(np.array([self.vector]))[0]
        return self.QV
    """
            
    def step(self, action):
        raise NotImplementedError           # returns tuple (new State object (can be this object updated), reward)
        
    @property
    def final(self):
        return self.isFinalState()

    def isFinalState(self):
        raise NotImplementedError
    
    @property
    def vector(self):
        if self.Vector is None: self.Vector = self.vectorize()
        return self.Vector
        
    def vectorize(self):
        raise NotImplementedError
        
class QNet(object):
    
    def compute(self, xbatch, reset_state = True):
        raise NotImplementedError

    def reset_state(self):
        pass
        
    __call__ = compute
    
    def train(self, xbatch, ybatch, eta, recompute = True):
        raise NotImplementedError
        
    def save(self, filename):
        pass

def masked_error(args):
    import keras.backend as K
    from keras import losses
    y_true, y_pred, mask = args
    loss = K.square(y_true-y_pred)
    return K.sum(loss*mask, axis=-1)

class QNetKeras(QNet):
    
    def __init__(self, keras_model):
        print "QNetKeras constructor"
        QNet.__init__(self)
        self.Model = keras_model

        # build rainable model
        
        y_pred = self.Model.output
        nactinos = self.Model.output.shape[-1]
        y_true = Input(name='y_true', shape=y_pred.shape)
        mask = Input(name='mask', shape=y_pred.shape)
        loss_out = Lambda(masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        ins = [self.Model.input] if type(self.Model.input) is not list else self.Model.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        #combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses)   #, metrics=combined_metrics)
        self.TrainableModel = trainable_model
        
        
        
    def compute(self, xbatch, reset_state = True):
        #print xbatch
        if reset_state:
            self.Model.reset_states()
        return self.Model.predict(xbatch)
        
    __call__ = compute
        
    def train(self, x, y_true, y_mask):
        return self.TrainableModel.train_on_batch([x, y_true, y_mask], ybatch)
        
    def loss(self, y, x):
        return self.Model.evaluate(x=x, y=y)
        
class AgentBase:
    
    def __init__(self, env, memory_size = 10000, qnet = None, 
                sample_size = 100, samples = 5, games_between_updates = 20, learns_between_updates = 1000,
                short_term_memory_size = 50, gamma = 0.9, eta = 0.1, 
                epsilon = 0.5, depsilon = 1.0e-5, min_epsilon = 0.01):
        self.Env = env
        self.Memory = ReplayMemory(memory_size)
        self.QNet = qnet or self.defaultNetwork(env)
        self.UpdateSamples = samples
        self.SampleSize = sample_size
        self.Gamma = gamma
        self.Eta = eta
        self.Epsilon = epsilon
        self.DEpsilon = depsilon
        self.MinEpsilon = min_epsilon
        self.Loss = None
        self.SmoothedLoss = None
        self.GamesBetweenUpdates = games_between_updates
        self.LearnsBetweenUpdates = learns_between_updates
        self.reset_update_counts()
        
        # old state info, saved by action()
        self.StateVector = None
        self.QVector = None
        self.QRMS = None
        
        #print "Agent created"

    def qvector(self, state):
        raise NotImplementedError
        
    def choice__(self, qs, epsilon, valid = None):
        f = 1.0/max(epsilon, 0.0001) - 1.0
        if valid is None:
            valid = np.arange(len(qs))
        qs = qs[list(valid)]
        qrms = math.sqrt(np.mean(np.square(qs)))
        if self.QRMS is None:
            self.QRMS = qrms
        else:
            self.QRMS = self.QRMS*0.9 + qrms*0.1
        x = qs/self.QRMS*f
        x -= np.max(x)
        e = np.exp(x)
        probs = e/np.sum(e)
        #print "choice: epsilon=",epsilon, "  valid=", valid, "  qs=",qs, "  probs=",probs
        return np.random.choice(valid, p=probs)
        
    def choice(self, qs, epsilon, valid=None):
        if epsilon > random.random():
            if valid is None:
                return random.randint(0, len(qs)-1)
            else:
                return random.choice(valid)
        else:
            if valid is None:
                return np.argmax(qs)
            else:
                return valid[np.argmax(qs[valid])]
        
    def updateLoss(self, loss):
        #print "Update loss(%s)" % (loss,)
        if self.SmoothedLoss is None:
            self.SmoothedLoss = loss
        else:
            self.SmoothedLoss = loss*0.1 + self.SmoothedLoss*0.9
        self.Loss = loss
        
    def defaultNetwork(self, env):
        raise NotImplementedError
        
    def reset(self):
        self.QNet.resetState()

    def action(self, state, epsilon=None):
        raise NotImplementedError
        
    def learn(self, action, reward, new_state, log=False):
        raise NotImplementedError
            
    def update(self, log=False):
        raise NotImplementedError

    def end(self):
        pass
        
    def reset_update_counts(self):
        self.GamesToUpdate = self.GamesBetweenUpdates
        self.LearnsToUpdate = self.LearnsBetweenUpdates
        
    def updateIfNeeded(self, learn = 0, game = 0):
        #print "updateIfNeeded"
        self.GamesToUpdate -= game
        self.LearnsToUpdate -= learn
        if self.Memory.size() >= self.SampleSize and \
            (self.GamesToUpdate <= 0 or self.LearnsToUpdate <= 0):
                self.update()

class AgentRNN(AgentBase):
    
    def reset(self):
        self.QNet.resetState()
        self.Game = []
        
    def qvector(self, state):
        v = state.vector.reshape((1,1,-1))
        qvector = self.QNet.compute(x, reset_state = False)[0,-1]  
        return qvector      

    def action(self, state, epsilon=None):
        if epsilon is None: epsilon = self.Epsilon
        qvector = self.qvector(state)
        self.QVector = qvector
        a = self.choice(qvector, epsilon, state.actions)
        qa = qvector[a]
        self.StateVector = state.vector.copy()
        return a, qa
        
    def learn(self, action, reward, new_state, log=False):
        tup = (self.StateVector, action, reward, new_state.actions)
        self.Game.append(tup)
        
    def end(self):
        state_vectors, actions, rewards, new_state_actions = zip(*self.Game)
        #print "endGame: xv=", type(xv), len(xv), xv
        self.Memory.add((np.array(state_vectors), actions, rewards, new_state_actions))
        self.updateIfNeeded(game=1)
        

    def update(self, log=False):
        #print "update"
        self.reset_update_counts()
        games = self.Memory.sample(self.UpdateSamples)
        if not games:   return 
        #print "sample:", len(games)
        loss = 0.0
        nloss = 0
        for state_vectors, actions, rewards, new_state_action_sets in games:
            L = len(state_vectors)
            state_vectors = state_vectors.reshape((1,)+state_vectors.shape)    # 1 minibatch
            state_qs = self.QNet.compute(state_vectors, reset_state = True).copy()
            new_state_qs = state_qs[0, 1:]
            #print "update:       xv=", xv[0]
            #print "     initial qns=", qns[0]
            for t in range(L):
                action = actions[t]
                v = rewards[t]
                if t+1 < L:       # last row is final state with empty q
                    qvec = new_state_qs[t]
                    #print qvec, new_state_action_sets
                    qmax = max([qvec[aa] for aa in new_state_action_sets[t]])
                    v += self.Gamma*qmax
                loss += (state_qs[0, t, action] - v)**2
                nloss += 1
                #print "q[%s] %s -> %s" % (a, qns[0, i, a], v)
                state_qs[0, t, action] = v
            #print "      final qns=", qns[0]
            self.QNet.train(state_vectors, state_qs, self.Eta)
        if nloss > 0:  self.Loss = math.sqrt(loss/nloss)
        self.Epsilon = max(self.Epsilon*(1.0 - self.DEpsilon), self.MinEpsilon)
        #print "Updated. Loss=", self.Loss, "   epsilon=",self.Epsilon
        
    def defaultNetwork(self, env):
        state_size = env.StateVectorSize
        in_size = state_size + 2        # + action and reward
        out_size = env.NActions
        inp = InputLayer((None, in_size))
        hidden_size = (in_size + out_size) * 2
        w = 0.0
        r1 = Tanh(LSTM(inp, hidden_size, hidden_size, name="R1", weight_decay=w))
        r2 = Tanh(LSTM(r1, hidden_size, hidden_size, name="R2", weight_decay=w))
        out = Sigmoid(LSTM(r2, out_size, hidden_size, name="Out", weight_decay=w))
        return Model(inp, out, L2Regression(out), applier_class=AdaDeltaApplier)


    
class Agent(AgentBase):
    
    """
    def __init__(self, env, memory_size = 10000, qnet = None, 
                sample_size = 60, samples = 2, update_interval = 5,
                short_term_memory_size = 50, gamma = 0.9, eta = 0.1, 
                epsilon = 0.5, depsilon = 1.0e-5, min_epsilon = 0.01):
        AgentBase.__init__(self, env, memory_size = memory_size, qnet = qnet, 
                sample_size = sample_size, samples = samples, update_interval = update_interval,
                short_term_memory_size = short_term_memory_size, gamma = gamma, eta = eta, 
                epsilon = epsilon, depsilon = depsilon, min_epsilon = min_epsilon)
    """

    def reset(self):
        self.QNet.resetState()

    def updateLoss(self, loss):
        if self.SmoothedLoss is None:
            self.SmoothedLoss = loss
        else:
            self.SmoothedLoss = loss*0.1 + self.SmoothedLoss*0.9
        self.Loss = loss
        
    def defaultNetwork(self, env):
        in_size = env.StateVectorSize
        out_size = env.NActions
        inp = InputLayer((in_size,))
        w = 0.0
        hsize = (in_size + out_size) * 3
        h1 = Tanh(Linear(inp, hsize, name="H1", weight_decay=w))
        h2 = Tanh(Linear(h1, hsize, name="H2", weight_decay=w))
        out = Tanh(Linear(h2, out_size, name="Out", weight_decay=w))
        return Model(inp, out, L2Regression(out), applier_class=AdaDeltaApplier)

    def qvector(self, state):
        v = state.vector
        if isinstance(v, (tuple, list)):
            x = [vi.reshape((1,) + vi.shape) for vi in v]
        else:
            x = v.reshape((1,) + v.shape)              # [nbatch=1, width]    
        return self.QNet.compute(x)[0]

    def action(self, state, epsilon=None):
        if epsilon is None: epsilon = self.Epsilon
        qvector = self.qvector(state)
        self.QVector = qvector
        self.StateVector = state.vector
        a = self.choice(qvector, epsilon, state.actions)
        qa = qvector[a]
        return a, qa
        
    def learn(self, action, reward, new_state, log=False):
        t = time.time()
        tup = (self.StateVector, action, reward, new_state.vector, new_state.final, new_state.actions, t)
        #if new_state.final:
        #print "added tuple:", self.LastState, " a:", self.LastAction, " r:", reward, " ns:", new_state, " t:", t
        if new_state.final or random.random() < 1.0:
            self.Memory.add(tup)
            if log:
                #print "added tuple:", tup
                pass
        self.updateIfNeeded(learn=1)

    def end(self):
        #print "calling updateIfNeeded"
        self.updateIfNeeded(game=1)
        
    def formatBatchX(self, x):
        # x is list of State.vector's, which can be either list of ndarray's or
        # list of tuples with ndarray's
        #
        # returns ndarray[batch, width]
        #
        
        if len(x) == 0:
            return []
            
        if isinstance(x[0], tuple):
            #print "x[0] is tuple"
            xlst = [np.array(xi) for xi in zip(*x)]
            #print "xlst[0]=", xlst[0]
        else:
            xlst = [np.array(x)]    # the network accepts list of inputs
        
        #print len(xlst[0]), len(xlst[1])
        #print xlst[0]
        
        return xlst           
        
    def concatBatches(self, x1, x2):
        #
        # x in list of [input1, input2, ...]
        # each input is ndarray
        #
        assert len(x1) == len(x2)       # same number of input arrays
        out = []
        for i in xrange(len(x1)):
            #print "x1:", x1[i]
            #print "x2:", x2[i]
            out.append(np.concatenate((x1[i], x2[i])))
        return out
        
        
    def update(self, log=False, samples=None, sample_size=None):
        samples = samples or self.UpdateSamples
        sample_size = sample_size or self.SampleSize
        
        #log = random.random() < 0.1
            
        self.reset_update_counts()
        if log:
            logf = open("log.txt", "w")
        for _ in range(samples):
            if log and False: logf.write("sample-------------\n")
            sample = self.Memory.sample(sample_size)
            #sample.append(tup)
    
            nsample = len(sample)
            
            #print "sample:", sample[0][0]
            
            snx = [nsv for sv, a, r, nsv, nsf, nsa, t in sample]
            snx = self.formatBatchX(snx)
            sx = [sv for sv, a, r, nsv, nsf, nsa, t in sample]
            sx = self.formatBatchX(sx)
            #print snx
            
            #print sx.shape
            #print snx.shape

            x = self.concatBatches(sx, snx)
            q = self.QNet.compute(x)

            qs = q[:nsample]
            qns = q[nsample:]

            #qns = self.QNet.compute(snx)
            #qs = self.QNet.compute(sx)
            qsy = qs.copy()

            #print "qs before:", qs
            loss = 0.0
            for i, (sv, a, r, ns, nsf, nsa, t) in enumerate(sample):
                v = r
                qmax = 0.0
                if not nsf:
                    qmax = max([qns[i,aa] for aa in nsa])
                    #v += self.Gamma*qmax
                    #v = (1.0+v)*(1.0+self.Gamma*qmax) - 1.0
                    v += self.Gamma*qmax
                loss += (qs[i,a]-v)**2
                if log:
                    logf.write("q: %s a: %d r: %f fin: %s qn: %s qmax: %s v:%f t:%f\n" % 
                        (qs[i], a, r, nsf, qns[i], qmax, v, t))
                #print "update: qs[%d]: %.2f -> %.2f (r: %.2f, qmax: %.2f%s)" % (a, qsy[i,a], v, r, qmax,
                #            "--- new state is final ---" if nsf else "")
                qsy[i,a] = v
            #print "training:"
            #for i, x in enumerate(sx):
            #    a = sample[i][1]
            #    print x, a, qs[i], qsy[i]
            if log:
                l0 = self.QNet.loss(qsy, x=sx)
            
            self.QNet.train(sx, qsy, self.Eta, recompute=True)     # qs was just computed
            if log:
                qsy_after_update = self.QNet.compute(sx)
                l1 = self.QNet.loss(qsy, x=sx)
                logf.write("loss %s -> %s\n" % (l0, l1))
                #for xx, yy0, yyt, yy1 in zip(sx, qs, qsy, qsy_after_update):
                #    logf.write("x=%s     y0=%s     y_=%s    y1=%s\n" % (xx, yy0, yyt, yy1))
            self.updateLoss(math.sqrt(loss/len(sample)))
        self.Epsilon = max(self.Epsilon*(1.0 - self.DEpsilon), self.MinEpsilon)
        #print "updated"
        if log: logf.close()
            