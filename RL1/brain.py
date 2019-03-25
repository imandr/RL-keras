from models import DirectDiffModel

class Brain(object):
    
    def __init__(self, qmodel, model_type, policy, gamma, *params, **args):
        self.QModel = qmodel
        self.RLModel = self.create_rl_model(qmodel, model_type, gamma, *params, **args)
        self.Policy = policy
        self.TModel = self.RLModel.tmodel()
        
    def create_rl_model(self, qmodel, model_type, gamma, *params, **args):
        if model_type == "ddiff":
            return DirectDiffModel(qmodel, gamma, *params, **args)
        
    def tmodel(self):
        return self.TModel
        
    def q(self, obsrvation):
        return self.QModel.predict_on_batch([observation])[0]
        
    def action(self, observation):
        q = self.q(observation)
        a = self.Policy(q)
        return a, q

    def trainingData(self, o0, a, o1, r, f):
        return self.RLModel.training_data(o0, a, o1, r, f)
