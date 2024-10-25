"""
S2S Decoder model.  (c) 2021 Georgia Tech

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
import pdb
import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN", attention=False):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type
        self.attention = attention

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer based on the "model_type" argument.            #
        #          Supported types (strings): "RNN", "LSTM". Instantiate the        #
        #          appropriate layer for the specified model_type.                  #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #       5) If attention is True, A linear layer to downsize concatenation   #
        #           of context vector and input                                     #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        # 1. Embedding layer
        self.embedding = nn.Embedding(self.output_size, self.emb_size)
        
        # 2. Recurrent layer (RNN or LSTM based on model_type)
        if self.model_type == "RNN":
            self.rnn = nn.RNN(self.emb_size, self.decoder_hidden_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(self.emb_size, self.decoder_hidden_size, batch_first=True)
        
        # 3. Linear layer (decoder_hidden_size -> output_size)
        self.linear1 = nn.Linear(self.decoder_hidden_size, self.output_size)
        
        # 4. LogSoftmax activation (with dim specified)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        # 5. Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # 6. Attention linear layer (only if attention is used)
        if self.attention:
            self.attn_combine = nn.Linear(self.encoder_hidden_size + self.emb_size, self.emb_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################

        # Apply cosine similarity between hidden state and encoder outputs
        hidden = hidden.squeeze(0)  # (N, hidden_dim)
        encoder_outputs = encoder_outputs.permute(0, 2, 1)  # (N, hidden_dim, T)

        # Calculate cosine similarity
        dot_product = torch.bmm(hidden.unsqueeze(1), encoder_outputs).squeeze(1)  # (N, T)
        norm_hidden = torch.norm(hidden, dim=1, keepdim=True)  # (N, 1)
        norm_encoder_outputs = torch.norm(encoder_outputs, dim=1)  # (N, T)
        
        attention_prob = dot_product / (norm_hidden * norm_encoder_outputs + 1e-8)  # (N, T)
        attention_prob = torch.softmax(attention_prob, dim=1).unsqueeze(1)  # (N, 1, T)
        
        return attention_prob

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    
    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ Forward pass of the decoder
            Args:
                input (tensor): input token tensor, shape (batch_size, 1)
                hidden (tensor): previous hidden state, shape (1, batch_size, hidden_size) or (num_layers, batch_size, hidden_size)
                encoder_outputs (tensor, optional): encoder outputs for attention, shape (batch_size, seq_len, encoder_hidden_size)
                attention (bool, optional): whether to apply attention
            Returns:
                output (tensor): output token probabilities, shape (batch_size, output_size)
                hidden (tensor): updated hidden state
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       1) Apply the dropout to the embedding layer                         #
        #                                                                           #
        #       2) If attention is true, compute the attention probabilities and    #
        #       use them to do a weighted sum on the encoder_outputs to determine   #
        #       the context vector. The context vector is then concatenated with    #
        #       the output of the dropout layer and is fed into the linear layer    #
        #       you created in the init section. The output of this layer is fed    #
        #       as input vector to your recurrent layer. Refer to the diagram       #
        #       provided in the Jupyter notebook for further clarifications. note   #
        #       that attention is only applied to the hidden state of LSTM.         #
        #                                                                           #
        #       3) Apply linear layer and log-softmax activation to output tensor   #
        #       before returning it.                                                #
        #                                                                           #
        #       If model_type is LSTM, the hidden variable returns a tuple          #
        #       containing both the hidden state and the cell state of the LSTM.    #
        #############################################################################

        # 1. Embedding the input token and applying dropout
        if input.dim() == 1:
            input = input.unsqueeze(1)  # Shape: (batch_size, 1)
        embedding = self.embedding(input)  # Shape: (batch_size, 1, embedding_size)
        embedding = self.dropout(embedding)

        # 2. Apply attention if enabled
        if encoder_outputs is None:
            print("here")

        if self.attention:
            # Compute attention weights
            attention_weights = self.compute_attention(hidden[0] if self.model_type == "LSTM" else hidden, encoder_outputs)  # Shape: (batch_size, 1, seq_len)
            # Create context vector by applying attention weights to encoder outputs
            context_vector = torch.bmm(attention_weights, encoder_outputs)  # Shape: (batch_size, 1, encoder_hidden_size)
            # Concatenate context vector with embedding (context on left, embedding on right)
            combined = torch.cat((context_vector, embedding), dim=-1)  # Shape: (batch_size, 1, encoder_hidden_size + emb_size)
            # Combine context and embedding
            embedding = self.attn_combine(combined)  # Shape: (batch_size, 1, emb_size)


        # 3. Pass the embedding through the RNN or LSTM
        if self.model_type == "RNN":
            output, hidden = self.rnn(embedding, hidden)
        else:
            output, hidden = self.rnn(embedding, hidden)

        # 4. Flatten the output to remove extra dimension
        output = output.squeeze(1)  # Shape: (batch_size, hidden_size)

        # 5. Apply linear layer to convert hidden size to output size (vocab size)
        output = self.linear1(output)  # Shape: (batch_size, output_size)

        # 6. Apply softmax to get probabilities
        output = self.softmax(output)  # Shape: (batch_size, output_size)

        return output, hidden

        
        