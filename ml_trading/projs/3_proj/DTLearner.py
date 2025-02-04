"""

"""

import pdb
import numpy as np

class DTLearner(object):
    """
    
    """

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "kmccarville3"

    def __init__(self, leaf_size=1):
        """
        Constructor method
        """

        self.tree = None
        self.leaf_size = 1

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        """
        self.tree = self.build_tree(np.column_stack((data_x, data_y)))

    def best_split(self, x_data, y_data):
        """
        Finds the best split for the data
        """
        corr = np.corrcoef(x_data, y_data, rowvar=False)[:-1, -1]
        factor = np.argmax(np.abs(corr))
        split_val = np.median(x_data[:, factor])
        return factor, split_val

    def build_tree(self, data):
        """
        Builds the tree
        """
        pdb.set_trace()
        x_samp, y_samp = data[:, :-1], data[:, -1]

        if len(np.unique(y_samp)) == 1:
            return np.array([[self.tree, y_samp[0], np.nan, np.nan]])
        
        if x_samp.shape[0] <= self.leaf_size:
            return np.array([[self.tree, np.mean(y_samp), np.nan, np.nan]])
        
        factor, split_val = self.best_split(x_samp, y_samp)
        print(factor, split_val)
        left_tree = self.build_tree(data[data[:, factor] <= split_val])
        right_tree = self.build_tree(data[data[:, factor] > split_val])

        root = np.array(["x" + str(factor), split_val, 1, left_tree.shape[0] + 1])
        return np.vstack((np.vstack((root, left_tree)), right_tree))

    def query_tree(self, points):
        """
        Query the tree
        """
        result = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            row = self.tree[0]
            while row[0] != "leaf":
                if points[i, int(row[0][1:])] <= row[1]:
                    row = self.tree[int(row[2])]
                else:
                    row = self.tree[int(row[3])]
            result[i] = row[1]
        return result