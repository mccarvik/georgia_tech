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
import BagLearner as bl
import RTLearner as rtl
import DTLearner as dtl	 
import matplotlib.pyplot as plt 	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    if len(sys.argv) != 2:  		  	   		 	 	 			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 			  		 			     			  	 
        sys.exit(1)  		  	   		 	 	 			  		 			     			  	 
    inf = open(sys.argv[1])
    try:
        data = np.array(  		  	   		 	 	 			  		 			     			  	 
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]  		  	   		 	 	 			  		 			     			  	 
        )
    except Exception as e:
        inf = open(sys.argv[1])
        data = np.genfromtxt(inf, delimiter=",", skip_header=1)
        # Drop the first column and then cast to float
        data = np.delete(data, 0, axis=1).astype(float)
  		  	   		 	 	 			  		 			     			  	 
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
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  	
    # learner = dtl.DTLearner(leaf_size=1)  # create a DTLearner
    # learner.add_evidence(train_x, train_y)  # train it  		  	   		 	 	 			  		 			     			  	 
    # print(learner.author())

    # Experiment 1
    results = []
    for leaf_size in range(1, 51):
        learner = dtl.DTLearner(leaf_size=leaf_size)  # create a DTLearner
        learner.add_evidence(train_x, train_y)  # train it

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_in = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        c_in = np.corrcoef(pred_y, y=train_y)[0, 1]
        accuracy_in = np.mean((pred_y - train_y) ** 2)

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_out = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        c_out = np.corrcoef(pred_y, y=test_y)[0, 1]
        accuracy_out = np.mean((pred_y - test_y) ** 2)

        results.append((leaf_size, rmse_in, c_in, rmse_out, c_out, accuracy_in, accuracy_out))

    # Print the results in a chart format
    print("Leaf Size | In Sample RMSE | In Sample Corr | Out Sample RMSE | Out Sample Corr | In Sample Accuracy | Out Sample Accuracy")
    for result in results:
        print(f"{result[0]:9} | {result[1]:15.4f} | {result[2]:14.4f} | {result[3]:16.4f} | {result[4]:15.4f} | {result[5]:17.4f} | {result[6]:18.4f}")

    # Extract data for plotting
    leaf_sizes = [result[0] for result in results]
    accuracy_in_samples = [result[5] for result in results]
    accuracy_out_samples = [result[6] for result in results]

    # Plot RMSE for in-sample and out-of-sample
    plt.figure(figsize=(10, 6))
    rmse_in_samples = [result[1] for result in results]
    rmse_out_samples = [result[3] for result in results]
    plt.plot(leaf_sizes, rmse_in_samples, label='In Sample RMSE', marker='o')
    plt.plot(leaf_sizes, rmse_out_samples, label='Out Sample RMSE', marker='o')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Leaf Size')
    plt.legend()
    plt.grid(True)
    # plt.gca().invert_xaxis()  # Invert the x-axis to decrease from 50 to 1
    plt.savefig("exp1_rmse.png")


    # Bagging test
    # Experiment 2: Bag Learner with 10 bags
    # Experiment 2: Bag Learner with different number of bags
    bag_counts = [10, 20, 30, 40]
    for bags in bag_counts:
        results_bag = []
        for leaf_size in range(1, 51):
            learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": leaf_size}, bags=bags, boost=False, verbose=False)
            learner.add_evidence(train_x, train_y)  # train it

            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            rmse_in = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
            c_in = np.corrcoef(pred_y, y=train_y)[0, 1]
            accuracy_in = np.mean((pred_y - train_y) ** 2)

            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            rmse_out = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
            c_out = np.corrcoef(pred_y, y=test_y)[0, 1]
            accuracy_out = np.mean((pred_y - test_y) ** 2)
            results_bag.append((leaf_size, rmse_in, c_in, rmse_out, c_out, accuracy_in, accuracy_out))

        # Print the results in a chart format
        print(f"Results for {bags} bags")
        print("Leaf Size | In Sample RMSE | In Sample Corr | Out Sample RMSE | Out Sample Corr | In Sample Accuracy | Out Sample Accuracy")
        for result in results_bag:
            print(f"{result[0]:9} | {result[1]:15.4f} | {result[2]:14.4f} | {result[3]:16.4f} | {result[4]:15.4f} | {result[5]:17.4f} | {result[6]:18.4f}")

        # Extract data for plotting
        leaf_sizes = [result[0] for result in results_bag]
        rmse_in_samples = [result[1] for result in results_bag]
        rmse_out_samples = [result[3] for result in results_bag]

        # Plot RMSE for in-sample and out-of-sample
        plt.figure(figsize=(10, 6))
        plt.plot(leaf_sizes, rmse_in_samples, label='In Sample RMSE', marker='o')
        plt.plot(leaf_sizes, rmse_out_samples, label='Out Sample RMSE', marker='o')
        plt.xlabel('Leaf Size')
        plt.ylabel('RMSE')
        plt.title(f'Bag Learner RMSE vs Leaf Size ({bags} bags)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"exp2_rmse_{bags}_bags.png")


        # Expermient 3
        # Experiment 3: Compare DTLearner and RTLearner using MAE and MAPE

        results_dt = []
        results_rt = []
        for leaf_size in range(1, 51):
            # DTLearner
            learner_dt = dtl.DTLearner(leaf_size=leaf_size)
            learner_dt.add_evidence(train_x, train_y)

            # RTLearner
            learner_rt = rtl.RTLearner(leaf_size=leaf_size)
            learner_rt.add_evidence(train_x, train_y)

            # Evaluate in sample for DTLearner
            pred_y_dt_in = learner_dt.query(train_x)
            mae_dt_in = np.mean(np.abs(train_y - pred_y_dt_in))
            r2_dt_in = 1 - (np.sum((train_y - pred_y_dt_in) ** 2) / np.sum((train_y - np.mean(train_y)) ** 2))

            # Evaluate out of sample for DTLearner
            pred_y_dt_out = learner_dt.query(test_x)
            mae_dt_out = np.mean(np.abs(test_y - pred_y_dt_out))
            r2_dt_out = 1 - (np.sum((test_y - pred_y_dt_out) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2))

            results_dt.append((leaf_size, mae_dt_in, r2_dt_in, mae_dt_out, r2_dt_out))

            # Evaluate in sample for RTLearner
            pred_y_rt_in = learner_rt.query(train_x)
            mae_rt_in = np.mean(np.abs(train_y - pred_y_rt_in))
            r2_rt_in = 1 - (np.sum((train_y - pred_y_rt_in) ** 2) / np.sum((train_y - np.mean(train_y)) ** 2))

            # Evaluate out of sample for RTLearner
            pred_y_rt_out = learner_rt.query(test_x)
            mae_rt_out = np.mean(np.abs(test_y - pred_y_rt_out))
            r2_rt_out = 1 - (np.sum((test_y - pred_y_rt_out) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2))

            results_rt.append((leaf_size, mae_rt_in, r2_rt_in, mae_rt_out, r2_rt_out))

        # Extract data for plotting
        leaf_sizes = [result[0] for result in results_dt]
        mae_dt_in_samples = [result[1] for result in results_dt]
        r2_dt_in_samples = [result[2] for result in results_dt]
        mae_dt_out_samples = [result[3] for result in results_dt]
        r2_dt_out_samples = [result[4] for result in results_dt]

        mae_rt_in_samples = [result[1] for result in results_rt]
        r2_rt_in_samples = [result[2] for result in results_rt]
        mae_rt_out_samples = [result[3] for result in results_rt]
        r2_rt_out_samples = [result[4] for result in results_rt]

        # Plot MAE for in-sample
        plt.figure(figsize=(10, 6))
        plt.plot(leaf_sizes, mae_dt_in_samples, label='DTLearner In Sample MAE', marker='o')
        plt.plot(leaf_sizes, mae_rt_in_samples, label='RTLearner In Sample MAE', marker='o')
        plt.xlabel('Leaf Size')
        plt.ylabel('MAE')
        plt.title('In Sample MAE vs Leaf Size')
        plt.legend()
        plt.grid(True)
        plt.savefig("exp3_mae_in_sample.png")

        # Plot MAE for out-of-sample
        plt.figure(figsize=(10, 6))
        plt.plot(leaf_sizes, mae_dt_out_samples, label='DTLearner Out Sample MAE', marker='o')
        plt.plot(leaf_sizes, mae_rt_out_samples, label='RTLearner Out Sample MAE', marker='o')
        plt.xlabel('Leaf Size')
        plt.ylabel('MAE')
        plt.title('Out Sample MAE vs Leaf Size')
        plt.legend()
        plt.grid(True)
        plt.savefig("exp3_mae_out_sample.png")

        # Plot R^2 for in-sample
        plt.figure(figsize=(10, 6))
        plt.plot(leaf_sizes, r2_dt_in_samples, label='DTLearner In Sample R^2', marker='o')
        plt.plot(leaf_sizes, r2_rt_in_samples, label='RTLearner In Sample R^2', marker='o')
        plt.xlabel('Leaf Size')
        plt.ylabel('R^2')
        plt.title('In Sample R^2 vs Leaf Size')
        plt.legend()
        plt.grid(True)
        plt.savefig("exp3_r2_in_sample.png")

        # Plot R^2 for out-of-sample
        plt.figure(figsize=(10, 6))
        plt.plot(leaf_sizes, r2_dt_out_samples, label='DTLearner Out Sample R^2', marker='o')
        plt.plot(leaf_sizes, r2_rt_out_samples, label='RTLearner Out Sample R^2', marker='o')
        plt.xlabel('Leaf Size')
        plt.ylabel('R^2')
        plt.title('Out Sample R^2 vs Leaf Size')
        plt.legend()
        plt.grid(True)
        plt.savefig("exp3_r2_out_sample.png")


  		  	   		 	 	 			  		 			     			  	 
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
