"""
This file contains utility functions for the machine learning assignment.
"""

import matplotlib.pyplot as plt


def val_curve(values, output, model="", x_axis=""):
    """
    Plot a Validatuon curve
    """

    plt.plot(values, output)
    plt.title("Validation curve - {}".format(model))
    plt.xlabel(x_axis)
    plt.ylabel("Accuracy")
    plt.savefig("pngs/validation_curve_{}.png".format(model), dpi=300)
    plt.close()


def learn_curve(values, train_output, test_output, model="", x_axis="percentage of training data"):
    """
    Plot a Learning curve
    """
    plt.figure(figsize=(10, 6))
    # Plot the mean training scores
    plt.plot(values, train_output, label="Training score", color="darkorange", marker='o')
    # Plot the mean test scores
    plt.plot(values, test_output, label="Test Score", color="navy", marker='o')
    plt.legend(loc="best")
    plt.title("Learning curve - {}". format(model))
    plt.xlabel(x_axis)
    plt.ylabel("Accuracy")
    plt.savefig("pngs/learning_curve_{}.png".format(model), dpi=300)
    plt.close()