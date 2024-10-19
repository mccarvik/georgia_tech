"""
S2S Encoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden states of the Encoder(namely, Linear - ReLU - Linear).    #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        self.embedding = nn.Embedding(self.input_size,self.emb_size)
        if self.model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size, self.encoder_hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(self.emb_size, self.encoder_hidden_size, batch_first=True)

        self.linear1 = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
        self.activation_final = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the state coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #                                                                           #
        #       Do not apply any linear layers/Relu for the cell state when         #
        #       model_type is LSTM before returning it.                             #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        # input is (batch size, seq_len)
        # output is (batch size, seq len, embedding dim)
        embedding = self.embedding(input)
        # dropout certain signals. still output is (batch_size, seq len , embedding dim)
        embedding = self.dropout(embedding)
        # batch size dim is first.
        if self.model_type == "RNN":
            output,hidden= self.rnn(embedding)
        else:
            output, (hidden, cell) = self.rnn(embedding)
        # pass in hidden thru linear layers.
        hidden = self.linear1(hidden)
        hidden = self.activation(hidden)
        hidden = self.linear2(hidden)
        # apply final tanh activation
        hidden = self.activation_final(hidden)

        # hidden state will be used for decoder's first step. return the tuple for LSTM case.
        if self.model_type == "LSTM":
            hidden = (hidden, cell)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden
