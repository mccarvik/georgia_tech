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
    

    def build_tree(self, data):
        """
        Builds the tree
        """
        # if data is less than max leaf size
        if data.shape[0] <= self.leaf_size:
            return np.array([["leaf", np.mean(data[:, -1]), np.nan, np.nan]])
        
        # pick random factor
        factor = np.random.randint(0, data.shape[1] - 1)
        # split on it
        split_val = np.median(data[:, factor])
        # no change, make a leaf
        if np.all(data[:, factor] <= split_val):
            return np.array([["leaf", np.mean(data[:, -1]), np.nan, np.nan]])
        else:
            left_tree = self.build_tree(data[data[:, factor] <= split_val])
            right_tree = self.build_tree(data[data[:, factor] > split_val])
            root = np.array([[factor, split_val, 1, left_tree.shape[0] + 1]])
            return np.vstack((root, left_tree, right_tree))


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

    