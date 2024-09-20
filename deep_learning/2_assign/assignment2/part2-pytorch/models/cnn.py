"""
Vanilla CNN model.  (c) 2021 Georgia Tech

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
    print("Roger that from cnn.py!")


class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and no padding                           #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################

        # Define the input and output dimensions for the convolutional layer
        self.conv_input_dim = 3  # Input channels (e.g., RGB image has 3 channels)
        self.conv_output_dim = 32  # Number of output channels (filters)
        self.convInput_padding = 0  # No padding
        self.convkernel_dim = 7  # Kernel size (7x7)
        self.convkernel_stride = 1  # Stride of the convolution

        # Define the kernel size and stride for the max pooling layer
        self.maxPoolKernel_dim = 2  # Kernel size (2x2)
        self.maxPoolKernel_stride = 2  # Stride of the max pooling

        # Define the input and output dimensions for the fully connected layer
        self.fc_input_dim = 32 * 13 * 13  # Flattened dimension after conv and pooling
        self.fc_output_dim = 10  # Number of output classes

        # Initialize the convolutional layer
        self.conv2d = nn.Conv2d(self.conv_input_dim, self.conv_output_dim, self.convkernel_dim, 
                    stride=self.convkernel_stride, padding=self.convInput_padding)
        
        # Initialize the activation function
        self.activation = nn.ReLU()
        
        # Initialize the max pooling layer
        self.maxpool2d = nn.MaxPool2d(self.maxPoolKernel_dim, self.maxPoolKernel_stride, padding=0)
        
        # Initialize the fully connected layer
        self.fc1 = nn.Linear(self.fc_input_dim, self.fc_output_dim)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        # Apply the convolutional layer
        out = self.conv2d(x)
        # Apply the activation function
        out = self.activation(out)
        # Apply the max pooling layer
        out = self.maxpool2d(out)

        # Flatten the output from the convolutional and pooling layers
        flattened_dim = out.size()[1] * out.size()[2] * out.size()[3]
        flat_out = torch.reshape(out, (len(out), flattened_dim))
        
        # Apply the fully connected layer
        outs = self.fc1(flat_out)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
