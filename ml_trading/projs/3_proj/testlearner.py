""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import math  		  	   		 	 	 			  		 			     			  	 
import sys
import pdb  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dtl	  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    if len(sys.argv) != 2:  		  	   		 	 	 			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 			  		 			     			  	 
        sys.exit(1)  		  	   		 	 	 			  		 			     			  	 
    inf = open(sys.argv[1])
    inf.readline()  # skip the first header line
    data = np.array(  		  	   		 	 	 			  		 			     			  	 
        [list(map(float, s.strip().split(","))) for s in inf.readlines()]  		  	   		 	 	 			  		 			     			  	 
    )  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # compute how much of the data is training and testing  		  	   		 	 	 			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		 	 	 			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # separate out training and testing data  		  	   		 	 	 			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		 	 	 			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		 	 	 			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		 	 	 			  		 			     			  	 
    test_y = data[train_rows:, -1]  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		 	 	 			  		 			     			  	 
    print(f"{test_y.shape}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # create a learner and train it  		  	   		 	 	 			  		 			     			  	 
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  	
    learner = dtl.DTLearner(leaf_size=1)  # create a DTLearner
    learner.add_evidence(train_x, train_y)  # train it  		  	   		 	 	 			  		 			     			  	 
    print(learner.author())  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # evaluate in sample  		  	   		 	 	 			  		 			     			  	 
    pred_y = learner.query(train_x)  # get the predictions  		  	   		 	 	 			  		 			     			  	 
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print("In sample results")  		  	   		 	 	 			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		 	 	 			  		 			     			  	 
    c = np.corrcoef(pred_y, y=train_y)  		  	   		 	 	 			  		 			     			  	 
    print(f"corr: {c[0,1]}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # evaluate out of sample  		  	   		 	 	 			  		 			     			  	 
    pred_y = learner.query(test_x)  # get the predictions  		  	   		 	 	 			  		 			     			  	 
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print("Out of sample results")  		  	   		 	 	 			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		 	 	 			  		 			     			  	 
    c = np.corrcoef(pred_y, y=test_y)  		  	   		 	 	 			  		 			     			  	 
    print(f"corr: {c[0,1]}")  		  	   		 	 	 			  		 			     			  	 
