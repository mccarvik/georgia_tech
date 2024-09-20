"""
Two Layer Network Model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from twolayer.py!")


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super().__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################

        # Store the input dimension, hidden size, and number of classes
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # Define the first fully connected layer
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size, bias=True)
        # Define the sigmoid activation function
        self.sig = nn.Sigmoid()
        # Define the second fully connected layer
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes, bias=True)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        # Flatten the input tensor except for the batch dimension
        flattened_dim = x.size()[1] * x.size()[2] * x.size()[3]
        flat_x = torch.reshape(x, (len(x), flattened_dim))
        # Pass data through the first fully connected layer
        out = self.fc1(flat_x)
        # Apply the sigmoid activation function
        out = self.sig(out)
        # Pass data through the second fully connected layer
        out = self.fc2(out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
