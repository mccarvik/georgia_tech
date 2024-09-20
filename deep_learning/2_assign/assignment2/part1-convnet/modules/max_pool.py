"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = {}
        self.dx = None

    def _save_mask(self, x_slice, coords):
        """
        Save the mask of the max values for backpropagation.
        
        :param x_slice: The current slice of the input, (N, C, pool_height, pool_width)
        :param coords: The coordinates (i, j) in the output where this slice corresponds
        """
        # Initialize the mask with zeros
        mask = np.zeros_like(x_slice)
        # Get the shape of the slice
        batch_size, channels, pool_height, pool_width = x_slice.shape
        # Reshape the slice to (N, C, pool_height * pool_width) for easier manipulation
        x_flat = x_slice.reshape(batch_size, channels, pool_height * pool_width)
        # Find the index of the max value in each slice
        max_indices = np.argmax(x_flat, axis=2)
        # Create indices for batch and channel dimensions
        batch_indices, channel_indices = np.indices((batch_size, channels))
        # Set the mask at the max value indices to 1
        mask.reshape(batch_size, channels, pool_height * pool_width)[batch_indices, channel_indices, max_indices] = 1
        # Save the mask in the cache with the corresponding coordinates
        self.cache[coords] = mask


    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################

        batch_size, channels, height_in, width_in = x.shape
        pool_height, pool_width = self.kernel_size, self.kernel_size
        height_out = 1 + (height_in - pool_height) // self.stride
        width_out = 1 + (width_in - pool_width) // self.stride
        out = np.zeros((batch_size, channels, height_out, width_out))

        # Iterate over the output height
        for i in range(height_out):
            # Iterate over the output width
            for j in range(width_out):
                # Calculate the start and end indices for the current slice
                height_start = i * self.stride
                height_end = height_start + pool_height
                width_start = j * self.stride
                width_end = width_start + pool_width

                # Extract the current slice from the input
                x_slice = x[:, :, height_start:height_end, width_start:width_end]

                # Save the mask for backpropagation
                self._save_mask(x_slice, coords=(i, j))

                # Perform max pooling on the current slice
                out[:, :, i, j] = np.max(x_slice, axis=(2, 3))
        
        self.dim = (batch_size, channels, height_out, width_out)
        self.cache_x = np.array(x, copy=True)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        # Reshape dout to match the dimensions of the forward pass output
        dout = dout.reshape(self.dim)
        
        # Extract the shape of the output gradient
        batch_size, channels, output_height, output_width = dout.shape

        # Initialize the gradient of the input with zeros
        self.dx = np.zeros_like(self.cache_x)
        
        # Pooling dimensions
        pool_height, pool_width = self.kernel_size, self.kernel_size
        
        # Iterate over the output height
        for i in range(output_height):
            # Iterate over the output width
            for j in range(output_width):
                # Calculate the start and end indices for the current slice
                start_height = i * self.stride
                end_height = start_height + pool_height
                start_width = j * self.stride
                end_width = start_width + pool_width
                
                # Update the gradient of the input based on the upstream gradient and the mask
                self.dx[:, :, start_height:end_height, start_width:end_width] += dout[:, :, i:i+1, j:j+1]*self.cache[(i,j)]
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
