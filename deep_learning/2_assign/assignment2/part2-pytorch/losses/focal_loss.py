"""
Focal Loss Wrapper.  (c) 2021 Georgia Tech

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
import torch.nn.functional as F
import numpy as np


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from focal_loss.py!")


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """

    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################

    # Calculate the effective number of samples for each class
    effective_num = 1.0 - np.power(beta, cls_num_list)
    # Calculate the weights for each class based on the effective number
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    # Normalize the weights so that they sum to the number of classes
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    # Convert the weights to a PyTorch tensor
    per_cls_weights = torch.FloatTensor(per_cls_weights)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.0):
        super().__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################

        # Compute the cross-entropy loss without reduction
        cross_entropy = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        # Compute the probability of the correct class
        p = torch.exp(-cross_entropy)
        # Compute the focal loss
        loss = ((1 - p) ** self.gamma * cross_entropy).mean()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
