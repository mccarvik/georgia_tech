"""
A simple wrapper for Random Tree Learner
"""

import numpy as np

class RTLearner(object):
    """
    Random Tree Learner
    """

    def author(self):
        """
        Returns the author of this code
        """
        return "kmccarville3"


    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """

        self.tree = None
        self.verbose = verbose
        self.leaf_size = leaf_size
        # need this to check for no change
        # self.last_y = None


    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        """
        self.tree = self.build_tree(np.column_stack((data_x, data_y)))
    

    def query(self, points):
        """
        Query the tree based on the points input
        Same as DTLearner
        """
        result = np.zeros(points.shape[0])
        for ctr in range(points.shape[0]):
            node = 0
            while node < self.tree.shape[0]:
                if self.tree[node, 0] == "leaf":
                    result[ctr] = float(self.tree[node, 1])
                    break
                feature = int(self.tree[node, 0][1:])
                if points[ctr, feature] <= float(self.tree[node, 1]):
                    node += int(self.tree[node, 2])
                else:
                    node += int(self.tree[node, 3])
            result[ctr] = self.tree[node, 1]
        return result

    