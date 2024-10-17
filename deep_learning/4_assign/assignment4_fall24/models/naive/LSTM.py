"""
LSTM model.  (c) 2021 Georgia Tech

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

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        #i_t: input gate
        self.weight_ii = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.bias_ii = nn.Parameter(torch.zeros(self.hidden_size))
        self.weight_hi =  nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.bias_hi = nn.Parameter(torch.zeros(self.hidden_size))
        self.sigmoid_i = nn.Sigmoid()


        # f_t: the forget gate
        self.weight_if = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.bias_if =nn.Parameter(torch.zeros(self.hidden_size))
        self.weight_hf =nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.bias_hf = nn.Parameter(torch.zeros(self.hidden_size))
        self.sigmoid_f = nn.Sigmoid()



        # g_t: the cell gate
        self.weight_ig = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.bias_ig = nn.Parameter(torch.zeros(self.hidden_size))
        self.weight_hg = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.bias_hg = nn.Parameter(torch.zeros(self.hidden_size))
        self.tanh_g = nn.Tanh()

        
        
        # o_t: the output gate
        self.weight_io = nn.Parameter(torch.zeros(self.input_size, self.hidden_size))
        self.bias_io =nn.Parameter(torch.zeros(self.hidden_size))
        self.weight_ho = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        self.bias_ho = nn.Parameter(torch.zeros(self.hidden_size))
        self.sigmoid_o = nn.Sigmoid()

        self.tanh_last = nn.Tanh()
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        
        # x.size() = [4, 3, 2]
        ht = torch.zeros(self.hidden_size, self.hidden_size)
        gt = torch.zeros(self.hidden_size, self.hidden_size)
        ct = torch.zeros(self.hidden_size, self.hidden_size)

        for i in range(0, x.shape[1]):
            xx = x[:, i, :]

            x1 = torch.matmul(xx,self.weight_ii)
            x2 = self.bias_ii
            x3 = torch.matmul(ht, self.weight_hi)
            x4 = self.bias_hi
            it = x1 + x2 +x3+x4
            it = self.sigmoid_i(it)

            xx = x[:, i, :]
            x1 = torch.matmul(xx, self.weight_if)
            x2 = self.bias_if
            x3 = torch.matmul(ht, self.weight_hf)
            x4 = self.bias_hf
            ft = x1 + x2 + x3 + x4
            ft = self.sigmoid_f(ft)

            x1 = torch.matmul(xx, self.weight_ig)
            x2 = self.bias_ig
            x3 = torch.matmul(ht, self.weight_hg)
            x4 = self.bias_hg
            gt = x1 + x2 + x3 + x4
            gt = self.tanh_g(gt)

            x1 = torch.matmul(xx, self.weight_io)
            x2 = self.bias_io
            x3 = torch.matmul(ht , self.weight_ho)
            x4 = self.bias_ho
            ot = x1 + x2 + x3 + x4
            ot = self.sigmoid_o(ot)

            ct = ft * ct + it * gt
            ht = ot * self.tanh_last(ct)
        h_t = ht
        c_t = ct

        
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
