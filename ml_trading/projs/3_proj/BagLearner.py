"""

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

    def __init__(self, models, kwargs={}, bags=1, boost=False, verbose=False):
        """
        
        """
        self.models = []
        self.bags = bags
        self.kwargs = kwargs
        self.boost = boost
        self.verbose = verbose
        
        # initialize each learner
        for i in range(bags):
            self.models.append(models(**kwargs))

        
    def add_evidence(self, data_s, data_y):
        """
        
        """

        for model in self.models:
            pass

    
    def query(self, points):
        """
        
        """
        y_hat = [models.query(points) for models in self.models]
        return np.mean(y_hat)
    
