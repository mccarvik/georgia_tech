"""
This file contains utility functions for the machine learning assignment.
"""

import matplotlib.pyplot as plt


def val_curve(values, output, model=""):
    """
    Plot a learning curve
    """

    if model == "knn":
        x_axis = "value for K"

    plt.plot(values, output)
    plt.title("Learning curve - {}".format(model))
    plt.xlabel(x_axis)
    plt.ylabel("Accuracy")
    plt.savefig("pngs/learning_curve_{}.png".format(model), dpi=300)
    plt.close()

def learn_curve(values, output, model=""):
    """
    Plot a validation curve
    """

    if model == "knn" or model == "svm":
        x_axis = "percentage of training data"
    

    plt.plot(values, output)
    plt.title("Validation curve - {}". format(model))
    plt.xlabel(x_axis)
    plt.ylabel("Accuracy")
    plt.savefig("pngs/validation_curve_{}.png".format(model), dpi=300)
    plt.close()