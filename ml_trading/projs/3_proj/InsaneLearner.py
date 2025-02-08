import numpy as np
import LinRegLearner as lin_reg
import BagLearner as bag_learn
class InsaneLearner(object):
    def author(self):
        return "kmccarville3"
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.models = []
        for i in range(20):
            self.models.append(bag_learn.BagLearner(learner=lin_reg.LinRegLearner, bags=20))
    def add_evidence(self, x_data, y_data):
        for model in range(20):
            self.models[model].add_evidence(x_data, y_data)
    def query(self, x_data):
        yhats = np.zeros([len(x_data), 20])
        for model_num in range(len(self.models)):
            yhats[:,model_num] == self.models[model_num].query(x_data)
        return np.mean(yhats, axis=1)