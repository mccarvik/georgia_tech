"""
A Bagging-based ensemble learner.
"""

import numpy as np
import pdb

class BagLearner(object):
    """
    
    """
    
    def author(self):
        """
        Returns the author of this code
        """
        return "kmccarville3"

    def __init__(self, learner, kwargs={}, bags=1, boost=False, verbose=False):
        """
        constructor
        """
        self.models = []
        self.bags = bags
        self.kwargs = kwargs
        self.boost = boost
        self.verbose = verbose
        
        # initialize each learner
        for i in range(bags):
            self.models.append(learner(**kwargs))

        
    def add_evidence(self, x_data, y_data):
        """
        Trains the ensemble of models using bootstrap aggregating (bagging).
        """
        # cycle through each learner
        for model in self.models:
            samples = np.random.choice(x_data.shape[0], x_data.shape[0], replace=True)
            # random learning
            x_bag = x_data[samples]
            y_bag = y_data[samples]
            model.add_evidence(x_bag, y_bag)

    
    def query(self, points):
        """
        Predicts the output for the given input points using the ensemble of models.
        """
        y_hat = [models.query(points) for models in self.models]
        return np.mean(y_hat, axis=0)
    
