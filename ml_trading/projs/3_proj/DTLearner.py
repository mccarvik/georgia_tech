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

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """

        self.tree = None
        self.verbose = verbose
        self.leaf_size = leaf_size
        # need this to check for no change
        self.last_y = None

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        """
        self.tree = self.build_tree(np.column_stack((data_x, data_y)))

    def best_split(self, x_data, y_data):
        """
        Finds the best split for the data
        """
        # corr = np.corrcoef(x_data, y_data, rowvar=False)[:-1, -1]
        # factor = np.argmax(np.abs(corr))
        # split_val = np.median(x_data[:, factor])
        # return factor, split_val
        
        # loop thru each factor
        max_correlation = np.NINF
        max_feature = np.NINF
        for feat in range(x_data.shape[1]):
            # find the correlation between the feature and the y data
            samp_corr = abs(np.corrcoef(x_data[:, feat], y_data, rowvar=False)[1, 0])
            if samp_corr > max_correlation:
                max_correlation = samp_corr
                max_feature = feat
        return max_feature, np.median(x_data[:, max_feature])

    
    def check_no_change(self, y_data):
        """
        Checks if the y data has changed
        """
        if self.last_y is None:
            self.last_y = y_data
            return False
        if np.array_equal(self.last_y, y_data):
            self.last_y = None
            return True
        self.last_y = y_data
        return False


    def build_tree(self, data):
        """
        Builds the tree
        """
        x_samp, y_samp = data[:, :-1], data[:, -1]

        if len(np.unique(y_samp)) == 1:
            return np.array([["leaf", y_samp[0], np.nan, np.nan]])
        
        if x_samp.shape[0] <= self.leaf_size or self.check_no_change(y_samp):
            return np.array([["leaf", np.mean(y_samp), np.nan, np.nan]])
        
        
        # pdb.set_trace()
        factor, split_val = self.best_split(x_samp, y_samp)
        # if split_val == 0 and factor == 0:
        #     pdb.set_trace()
        #     pass
        # print(factor, split_val)
        left_tree = self.build_tree(data[data[:, factor] <= split_val])
        right_tree = self.build_tree(data[data[:, factor] > split_val])

        root = np.array(["x" + str(factor), split_val, 1, left_tree.shape[0] + 1])
        return np.vstack((np.vstack((root, left_tree)), right_tree))

    def query(self, points):
        """
        Query the tree based on the points input
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
