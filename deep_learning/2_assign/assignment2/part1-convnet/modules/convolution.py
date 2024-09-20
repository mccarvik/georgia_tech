"""
2d Convolution Module.  (c) 2021 Georgia Tech

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

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        # Get the dimensions of the input and the filters
        n_size, colors, h_input, w_input = x.shape  # batch_size, color channels, height of input volume, width of input volume
        n_filts, n_inputc, h_fitlts, w_filts = self.weight.shape  # number of filters, number of in channels, height of filter volume, width of filter volume

        # Calculate the dimensions of the output volume given the padding and stride calc
        h_out = (h_input - h_fitlts + 2 * self.padding) // self.stride + 1  # height of output volume
        w_out = (w_input - w_filts + 2 * self.padding) // self.stride + 1  # width of output volume

        # Apply padding to the input volume
        pad = (self.padding, self.padding)
        x_pad = np.pad(array=x, pad_width=((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), mode='constant')

        # Initialize the output volume
        output_shape = n_size, n_filts, h_out, w_out
        out_weight = np.zeros(output_shape)

        # Perform the convolution operation
        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * self.stride, j * self.stride  # starting indices for the current slice
                h_end, w_end = h_start + h_fitlts, w_start + w_filts  # ending indices for the current slice

                # Convolve the filter with the current slice of the input volume
                out_weight[:, :, i, j] = np.sum(
                    x_pad[:, np.newaxis, :, h_start:h_end, w_start:w_end] * self.weight[np.newaxis, :, :, :],
                    axis=(2, 3, 4)
                )

        # Add the bias to the output volume
        out_bias = self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        out = out_weight + out_bias

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################

        # Extract dimensions of dout and x
        batch_size, c_out, h_out, w_out = dout.shape
        n, n_chans, h_in, w_in = x.shape
        n_filts, n_inputc, h_filts, w_filts = self.weight.shape

        # Apply padding to the input volume
        pad = (self.padding, self.padding)
        x_pad = np.pad(array=x, pad_width=((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), mode='constant')

        # Initialize gradients for bias, weights, and input
        self.db = dout.sum(axis=(0, 2, 3))  # Gradient of bias
        self.dw = np.zeros_like(self.weight)  # Gradient of weights
        self.dx = np.zeros_like(x_pad)  # Gradient of input

        # Perform the convolution backward pass
        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * self.stride, j * self.stride  # starting indices for the current slice
                h_end, w_end = h_start + h_filts, w_start + w_filts  # ending indices for the current slice

                # Compute gradient of input
                self.dx[:, :, h_start:h_end, w_start:w_end] += np.sum(
                    self.weight[np.newaxis, :, :, :] * dout[:, :, i:i+1, j:j+1, np.newaxis], axis=1
                )

                # Compute gradient of weights
                self.dw += np.sum(
                    x_pad[:, np.newaxis, :, h_start:h_end, w_start:w_end] * dout[:, :, i:i+1, j:j+1, np.newaxis], axis=0
                )

        # Remove padding from the gradient of input
        self.dx = self.dx[:, :, pad[0]:pad[0]+h_in, pad[1]:pad[1]+w_in]

        #############################################################################

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
