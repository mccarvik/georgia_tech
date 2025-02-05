"""

"""

import numpy as np
import pdb
import LinRegLearner as lin_reg
import BagLearner as bag_learn

class InsaneLearner(object):
    """
    
    """

    def author(self):
        """
        Returns the author of this code
        """
        return "kmccarville3"


    def __init__(self, verbose=False):
        """
        
        """
        self.verbose = verbose
        self.models = []

        # create 20 lin reg learner bag
        for i in range(20):
            self.models.append(bag_learn.BagLearner(learner=lin_reg.LinRegLearner, bags=20))
        

    def add_evidence(self, x_data, y_data):
        """
        
        """
        for model in range(20):
            self.models[model].add_evidence(x_data. y_data)

    def query(self, x_data):
        """
        
        """
        yhats = np.zeros([len(x_data), 20])
        ctr = 0
        # cycle through each model to get the prediction
        for model in self.models:
            yhats[:,ctr] == model.query(x_data)
            ctr += 1
        
        # return the mean prediction
        return np.mean(yhats, axis=1)
