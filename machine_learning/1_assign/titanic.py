"""
Assignment 1 - main run method for titanic dataset
"""

import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import learn_curve, val_curve
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle


DATA_PATH = "data/titanic/titanic3.xls"
RANDOM_SEED = 42 # the answer to everything


def run():
    """
    main run method for the script
    """
    train_df = pd.read_excel(DATA_PATH)
    train_df = preprocess(train_df)
    train_df, test_df = train_test_split(train_df, test_size=0.15, random_state=RANDOM_SEED)
    # knearest(train_df, test_df)
    support_vector_machine(train_df, test_df)


def support_vector_machine(train_df, test_df):
    """
    support vector machine
    """
    y_train = train_df["survived"]
    x_train = train_df.drop(["survived"], axis=1)
    y_test = test_df["survived"]
    x_test = test_df.drop(["survived"], axis=1)


    hyper = False
    if hyper:
        # Run through the kernel functions and get validation curves for each hyperparameter
        for kern_func in ["linear", "poly", "rbf", "sigmoid"]:

            # Define the hyperparameter values to be tested
            param_range = np.logspace(-3, 6, 10)
            svm_model = SVC(kernel=kern_func)

            # Create a validation curve
            train_scores, test_scores = validation_curve(
                svm_model, x_train, y_train, param_name="C", param_range=param_range,
                cv=3, scoring="accuracy", n_jobs=-1
            )
            # Plot the validation curves
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.figure(figsize=(10, 6))
            plt.title("Validation Curve for SVM - {}".format(kern_func))
            plt.xlabel("C Parameter")
            plt.ylabel("Accuracy")
            plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=2)
            plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=2)
            plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=2)
            plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=2)
            plt.legend(loc="best")
            plt.savefig("pngs/validation_curve_svm_{}.png".format(kern_func), dpi=300)
            plt.close()

    
    # run through the kernel functions (with best C - parameter) for amount of training data learning curve
    # Shuffle the data without replacement - Learning curve
    val_curve = True
    if val_curve:
        svm_model = SVC(kernel="poly", C=100000)
        x_accs = {}
        for x_perc in range(1,11):
            x_perc = x_perc / 10
            x_percent_index = int(x_perc * len(x_train))
            # Take the first X% of the shuffled data
            first_x_percent = x_train[:x_percent_index]
            y_train_iter = y_train[:x_percent_index]
            svm_model = SVC(kernel="poly", C=100000)
            svm_model.fit(first_x_percent, y_train_iter)
            y_pred = svm_model.predict(x_test)
            x_accs[x_perc*10] = accuracy_score(y_test, y_pred)
            print("Perc val: {}   acc: {}".format(int(x_perc*10), x_accs[x_perc*10]))
        learn_curve(list(x_accs.keys()), list(x_accs.values()), "svm")
   
    
    # run through the best kernel functions and best hyperparameter for iteration learning curve
    num_epochs = 10  # Set the number of epochs or iterations


def knearest(train_df, test_df):
    """
    k nearest neighbors
    """
    y_train = train_df["survived"]
    x_train = train_df.drop(["survived"], axis=1)
    y_test = test_df["survived"]
    x_test = test_df.drop(["survived"], axis=1)

    # Curve based on value of K - Validation curve
    k_accs = {}
    for k_val in range(1,10):
        knn = KNeighborsClassifier(n_neighbors=k_val)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        k_accs[k_val] = accuracy_score(y_test, y_pred)
        print("K val: {}   acc: {}".format(k_val, k_accs[k_val]))
    val_curve(list(k_accs.keys()), list(k_accs.values()), "knn")

    
    # Shuffle the data without replacement - Learning curve
    x_accs = {}
    for x_perc in range(1,11):
        x_perc = x_perc / 10
        # shuffled_data = shuffle(x_train, random_state=RANDOM_SEED)
        x_percent_index = int(x_perc * len(x_train))
        # Take the first X% of the shuffled data
        first_x_percent = x_train[:x_percent_index]
        y_train_iter = y_train[:x_percent_index]
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(first_x_percent, y_train_iter)
        y_pred = knn.predict(x_test)
        x_accs[x_perc*10] = accuracy_score(y_test, y_pred)
        print("K val: {}   acc: {}".format(int(x_perc*10), x_accs[x_perc*10]))
    learn_curve(list(x_accs.keys()), list(x_accs.values()), "knn")


def preprocess(train_df):
    """
    preprocess the data
    """
    # Drop meaningless columns
    train_df = train_df.drop(["name", "ticket", "cabin", 'body', 'boat', 'home.dest'], axis=1)
    train_df['sex'] = np.where(train_df['sex'] == 'male', True, False)
    # train_df = train_df.dropna()
    train_df = pd.get_dummies(train_df)

    # Count the missing values for each column
    missing_count = train_df.isnull().sum()
    # Display the count of missing values

    # age is the only column with missing values
    # we will try to replace the missing values with the mean age
    # also will try removing them though it is a lot but there is a chance the
    # reason these are missing are not correlated to the mean, ie they are missing 
    # from the elderly or the young
    # mean and median very close
    imputer = SimpleImputer(strategy='median')  # or 'median', 'mode'
    train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    return train_df


if __name__ == "__main__":
    run()