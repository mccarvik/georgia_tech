"""
MyModel model.  (c) 2021 Georgia Tech

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
import torch.nn.functional as functorch


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(128)
        self.batch3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout1 = nn.Dropout(p = 0.05)
        self.dropout2 = nn.Dropout(p = 0.1)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        # Apply first convolution, batch normalization, and ReLU activation
        x_conv_relu1 = functorch.relu(self.batch1(self.conv1(x))) # 10,32,32,32
        # Apply first max pooling
        x_pool1 = functorch.max_pool2d(x_conv_relu1, kernel_size = 2, stride = 2) # 10,32,16,16
        # Apply second convolution and ReLU activation
        x_conv_relu2 = functorch.relu(self.conv2(x_pool1)) # 10,64,16,16
        # Apply second max pooling
        x_pool2 = functorch.max_pool2d(x_conv_relu2, kernel_size = 2, stride = 2) # 10,64,8,8
        # Apply third convolution, batch normalization, and ReLU activation
        x_conv_relu3 = functorch.relu(self.batch2(self.conv3(x_pool2))) # 10,128,8,8
        # Apply third max pooling
        x_pool3 = functorch.max_pool2d(x_conv_relu3, kernel_size = 2, stride = 2) # 10,128,4,4
        # Apply dropout
        x_drop1 = self.dropout1(x_pool3) # 10,128,4,4
        # Apply fourth convolution, batch normalization, and ReLU activation
        x_conv_relu4 = functorch.relu(self.batch3(self.conv4(x_drop1))) # 10,256,4,4
        # Flatten the tensor
        x_trans = x_conv_relu4.view(-1, 256*4*4)
        # Apply dropout
        x_fcdrop1 = self.dropout2(x_trans)
        # Apply first fully connected layer and ReLU activation
        x_fc1 = functorch.relu(self.fc1(x_fcdrop1))
        # Apply second fully connected layer and ReLU activation
        x_fc2 = functorch.relu(self.fc2(x_fc1))
        # Apply dropout
        x_fcdrop2 = self.dropout2(x_fc2)
        # Apply third fully connected layer (output layer)
        x_fc3 = self.fc3(x_fcdrop2)
        # Set the output
        outs = x_fc3

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
