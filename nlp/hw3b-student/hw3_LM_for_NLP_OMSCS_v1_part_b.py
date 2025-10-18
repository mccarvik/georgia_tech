#!/usr/bin/env python
# coding: utf-8

# # Language Modeling
# 
# A language model attempts to approximate the underlying statistics of a text corpus $P(tok_n | tok_1, tok_2, ..., tok_{n-1}; \theta)$ where $\theta$ is a set of learned parameters/weights. For the purposes of this notebook, tokens will be words. Language models can be used for a variety of applications, one of which being text generation. In this assignement we will be looking at language modeling techniques of increasing sophistication.
# 
# **Tips:**
# - Read all the code. We don't ask you to write the training loops, evaluation loops, and generation loops, but it is often instructive to see how the models are trained and evaluated.
# - If you have a model that is learning (loss is decreasing), but you want to increase accuracy, try using ``nn.Dropout`` layers just before the final linear layer to force the model to handle missing or unfamiliar data.

# In[1]:


# start time - notebook execution
import time
start_nb = time.time()


# # Set up

# Import packages

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import unicodedata

# ignore all warnings
import warnings
warnings.filterwarnings('ignore')


# # Initialize the Autograder

# In[3]:


# import the autograder tests
import hw3b_tests as ag


# We will build a *vocabulary*, which will act as a dictionary of all the words our systems will know about. It will also allow us to map words to tokens, which will be unique indexes in the vocabulary. This will further allow us to transform words into one-hot vectors, where a word is represented as a vector of the same length as the vocabulary wherein all values are zeros except for the *i*th element, where *i* is the token number of the word.

# In[4]:


class Vocab:
    def __init__(self, name):
        self.name = name                             # The name of the vocabulary
        self._word2index = {}                        # Map words to token index
        self._word2count = {}                        # Track how many times a word occurs in a corpus
        self._index2word = {0: "SOS", 1: "EOS"}      # Map token indexs back into words
        self._n_words = 2 # Count SOS and EOS        # Number of unique words in the corpus

    # Get a list of all words
    def get_words(self):
      return list(self._word2count.keys())

    # Get the number of words
    def num_words(self):
      return self._n_words

    # Convert a word into a token index
    def word2index(self, word):
      return self._word2index[word]

    # Convert a token into a word
    def index2word(self, word):
      return self._index2word[word]

    # Get the number of times a word occurs
    def word2count(self, word):
      return self._word2count[word]

    # Add all the words in a sentence to the vocabulary
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    # Add a single word to the vocabulary
    def add_word(self, word):
        if word not in self._word2index:
            self._word2index[word] = self._n_words
            self._word2count[word] = 1
            self._index2word[self._n_words] = word
            self._n_words += 1
        else:
            self._word2count[word] += 1


# These are some helper functions to *normalize* texts, ie, make the text regular and remove some of the more problematic exceptions found in texts. This normalizer will make all words lowercase, trim plurals, and remove non-letter characters.

# In[5]:


# Convert any unicode to ascii
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Download a corpus. This corpus is the ascii text of the book, *The Silmarillion*, by J.R.R. Tolkein. It has a lot of non-common words and names to illustrate how language models deal with such things.

# In[6]:


# if data.txt is not in the current directory, download it
if not os.path.isfile('data.txt'):
  get_ipython().system('wget -O data.txt https://www.dropbox.com/s/pgvn1n7t4sjxt8r/silmarillion?dl=1')


# Let's read in the data and take a look at it.

# In[7]:


filename = 'data.txt'
with open(filename, encoding='utf-8') as f:
  text = f.read()
text[:1000]


# Normalize the text and build the vocabulary

# In[8]:


normalized_text = normalize_string(text)
VOCAB = Vocab("text")
VOCAB.add_sentence(normalized_text)


# Make training and testing data splits.

# In[9]:


# Convert every word into a token and build a numpy array of tokens
encoded_text = np.array([VOCAB.word2index(word) for word in normalized_text.split()])
encoded_text = np.array([VOCAB.word2index(word) for word in normalized_text.split()], dtype=np.int64)

print("The first 100 tokens")
print(encoded_text[:100])
# get the validation and the training data
test_split = 0.1
test_idx = int(len(encoded_text) * (1 - test_split))
TRAIN = encoded_text[:test_idx]
TEST = encoded_text[test_idx:]
# Decrease the size of the training set to make the assignment more tractable
TRAIN = TRAIN[:len(TRAIN)//10]


# # LSTM (20 Points)
# 
# A more sophisticated version of an RNN is a Long Short-Term Memory network (or an LSTM). It learns to decided what should be kept in the hidden state and what should be removed from the hidden state. This allows it to make better hidden states and thus learn a more accurate probability distribution and be a better generator.
# 
# We will make two LSTMs. First, we will make a neural network that uses Pytorch's built in `nn.LSTMCell`. The second time, we will write an LSTM memory cell from scratch.
# 
# **Complete the following network with two or more LSTMCell layers.** The network will take two inputs in its forward function:
# - `x`: a sequence of words, represented as one-hots. The input should be a tensor of shape `1 x vocab_size` That is, each row is a one-hot (batch size is 1).
# - `hc` which is a tuple containing (hidden_state, cell_state).
# 
# The output of the forward function will be:
# - A sequence of output log probabilities. This output should be a tensor of shape `1 x vocab_size` where each row is a log probability distribution.
# - A tuple containing (hidden_state, cell_state).
# 
# The network should contain two our more LSTMCell modules. Send the one-hot into the first LSTMCell along with the original `hc`. Then send the resulting hidden state to the next higher LSTMCell *along with the initial `hc`*. Keep doing this until you get to the top of the stack of LSTMCells. Once you get to the top of the stack, use an affine transformation to expand to vocabular size and generate a log probability with a log softmax.
# 
# Forward should return the output log probabilities and a (hidden state, cell state) tuple.

# In[10]:


# build the model using the pytorch nn module
class MyLSTM(nn.ModuleList):
  def __init__(self, input_size, hidden_size, cell_type = nn.LSTMCell):
    super(MyLSTM, self).__init__()

    # init the parameters
    self.hidden_dim = hidden_size
    self.input_size = input_size

    ### Use the cell_type passed into the constructor as the type of LSTM cell module
    ### that is made. For the first part of the assignment, this will be the
    ### default nn.LSTMCell. For the second part, this will be the custom-written
    ### LSTM cell type.

    ### BEGIN SOLUTION
    # well go with just two layers and see if this gets there
    # input will be the one-hot input size into the given hidden stat sixe
    self.lstm_layer1 = cell_type(input_size, hidden_size)
    # layer 2, also cell_type is LSMCell from the signature, took me a second there
    self.lstm_layer2 = cell_type(hidden_size, hidden_size)

    # linear layer at the end here to get i tb ack to vocab size
    self.lin_layer_pre_softmax = nn.Linear(hidden_size, input_size)
    # and a standard softmax at the finish here
    self.softmax_finish = nn.LogSoftmax(dim=1)
    ### END SOLUTION

  def forward(self, x, hc):
    # Return values
    output = None
    hidden = None
    cell = None

    # Pass the hidden and the cell state from one lstm cell to the next one
    # we also feed the output of the first layer lstm cell at time step t to the second layer cell
    # init both layer cells with the zero hidden and zero cell states

    ### BEGIN SOLUTION
    # so now we all just connect it like previous assignments
    # x and hc comeing from the paramters
    hid_lstm1, lstm1 = self.lstm_layer1(x, hc)
    # same here for layer 2
    hid_lstm2, lstm2 = self.lstm_layer2(hid_lstm1, hc)
    # get it back to vocab dimensions
    lin_input = self.lin_layer_pre_softmax(hid_lstm2)
    output_sf_logs = self.softmax_finish(lin_input)
    # and now we gotta update the hid and cell from above
    hidden = hid_lstm2
    cell = lstm2
    # forgot about output var name:
    output = output_sf_logs
    ### END SOLUTION

    return output, (hidden.detach(), cell.detach())

  def init_hidden(self):
    # initialize the hidden state and the cell state to zeros
    return (torch.zeros(1, self.hidden_dim), # 1 is the batch size
            torch.zeros(1, self.hidden_dim)) # 1 is the batch size


# Let's build our LSTM

# In[ ]:


# It's ok to change this cell, however, you should not need to change it much (if at all) - note: certain changes may break the autograder, e.g., 
# increasing the size of the hidden layer could cause out of memory errors in the autograder and large numbers of epochs could cause autograder to time
# out (pay attention to the runtime of your notebook and the warnings that are printed out at the end of the notebook)
LSTM_HIDDEN_SIZE = 32
LSTM_NUM_EPOCHS = 3
LSTM_LEARNING_RATE = 0.01


# In[12]:


lstm = MyLSTM(VOCAB.num_words(), LSTM_HIDDEN_SIZE)
optimizer_lstm = optim.SGD(lstm.parameters(), lr=LSTM_LEARNING_RATE)
criterion_lstm = nn.NLLLoss()


# In[13]:


# student check - the following test must return a value of 3 to receive credit (5 pts)
ag.LSTM_check()


# In[14]:


# student check - the following test must return a value of 5 to receive credit (5 pts)
ag.unit_test_LSTM_structure()


# ## LSTM---Training

# Here is the training loop. Notice it uses `get_rnn_x_y()` from HW2.
# 

# In[15]:


def train_lstm(net, optimizer, criterion, num_epochs, data):
  epoch_losses = []
  scheduler = ExponentialLR(optimizer, gamma=0.9)
  net.train()
  for epoch in range(num_epochs):
    losses = []
    hc = net.init_hidden()
    for i in range(len(data)-1):
      x, y = ag.get_rnn_x_y(data, i, VOCAB.num_words())
      x = x.float()
      output, hc = net(x, hc)
      loss = criterion(output, y)
      losses.append(loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if i%1000 == 0:
        print('iter', i, 'loss', torch.stack(losses).mean().item())
    scheduler.step()
    print('epoch', epoch, 'loss', torch.stack(losses).mean().item())
    epoch_losses.append(torch.stack(losses).mean().item())
  return epoch_losses


# In[16]:


epoch_losses = train_lstm(lstm, optimizer_lstm, criterion_lstm, num_epochs=LSTM_NUM_EPOCHS, data=TRAIN)


# In[17]:


plt.figure(1)
plt.clf()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(len(epoch_losses)))
plt.plot(epoch_losses)


# You should see a curve that slopes down steeply at first and then levels out to some asymptotic minimum.

# ## LSTM---Testing

# Evaluation works the same as with the RNN.

# In[18]:


# student check - the following test must return a value less than 1000 to receive credit (10 pts)
ag.eval_lstm_1(max_perplexity=1000)


# ## LSTM---Generation

# Generation works the same as the RNN. In fact you will notice that we can use the `prep_hidden_state` and `generate_rnn` functions without modification.

# In[19]:


# It's ok to change this cell
LSTM_TEMPERATURE = 0.5


# In[20]:


# COPY YOUR HW3-A SOLUTION HERE
# copied from hw3a
def token2onehot(token, vocab_size = VOCAB.num_words()):
  one_hot = None
  ### BEGIN SOLUTION
  # so first lets get zeroes
  one_hot_vector = [0] * vocab_size
  # set our word guy
  one_hot_vector[token] = 1

  # one_hot = one_hot_vector
  # need to make it a tensor, think thats my problem?
  # one_hot = torch.tensor(one_hot_vector)
  one_hot = torch.tensor(one_hot_vector).unsqueeze(0)
  ### END SOLUTION
  return one_hot
    
# COPIED FROM HW3-A
def prep_hidden_state(tokenized_input, rnn, verbose=False):
  # Get an initial hidden state
  hidden_state = rnn.init_hidden()
  # Run the input prompt through the RNN to build up the hidden state.
  # Discard the outputs (we are not trying to make predictions) until we get to the end
  for token in tokenized_input:
    if verbose:
      print("current token:", token, VOCAB.index2word(token))
    # Get the one-hot for the current token
    x = token2onehot(token)
    x = x.float()
    # Run the current one-hot and hidden state through the RNN
    output, hidden_state = rnn(x, hidden_state)
    # Get the highest predicted token
    next_token = output.argmax().item()
    if verbose:
      print("predicted next token:", next_token, VOCAB.index2word(next_token), '\n')
  return hidden_state

def log_to_percentage_probs(log_probs):
  perc_probs = torch.exp(log_probs)
  return perc_probs

# COPY YOUR HW3-A SOLUTION HERE
# taken from 3a
def my_temperature_sample(log_probs, temperature=1.0):
  token = None
  ### BEGIN SOLUTION
  # apply temp scaling FIRST, what we might have screwed up last time
  probs_temp = log_probs / temperature
  # THEN get the percentage probs, use func from above again
  perc_probs = log_to_percentage_probs(probs_temp)

  # divide by temp
  numerator = perc_probs
  # get the sum of probs
  denominator = numerator.sum(dim=1, keepdim=True)
  val = numerator / denominator
  # use our multinomial again
  draws = 1
  token = torch.multinomial(val, draws).item()
  ### END SOLUTION
  return token

# COPIED FROM HW3-A
def generate_rnn(rnn, num_new_tokens, token, hidden_state, fn=lambda d:d.argmax().item(), verbose=False):
  # Keep generating more by feeding the predicted output back into the RNN as input
  # Start with the last token of the input prompt and the newly prepped hidden state
  if verbose:
    print("Generating continuation:\n")
  continuation = []
  for n in range(num_new_tokens):
    if verbose:
      print("current token:", token, VOCAB.index2word(token))
    # Get the one-hot for the current token
    x = token2onehot(token)
    x = x.float()
    # Run the current one-hot through the RNN
    output, hidden_state = rnn(x, hidden_state)
    # Predict the next token
    next_token = fn(output)
    if verbose:
      print("predicted next token:", next_token, VOCAB.index2word(next_token), '\n')
    # Remember the new token
    continuation.append(next_token)
    # update the current
    token = next_token
  return continuation

# Example input prompt:
input_prompt = "the First War began"
# How long should the continuation be?
num_new_tokens = 10

# Normalize the input
normalized_input = normalize_string(input_prompt)
# Tokenize the input
tokenized_input = [VOCAB.word2index(w) for w in normalized_input.split()]
print("input prompt:", input_prompt)
print("input tokens:", tokenized_input, '\n')

# Get the hidden state that represents the input prompt
print("Prepping hidden state:\n")
hidden_state = prep_hidden_state(tokenized_input, lstm, verbose=True)

# Generate the continuation. Use the argmax function to sample from the RNN's outputs
token = tokenized_input[-1]
continuation = generate_rnn(lstm, num_new_tokens, token, hidden_state, fn=lambda d:my_temperature_sample(d, LSTM_TEMPERATURE), verbose=True)

# All done
print("Final continuation:")
print(continuation)
continuation_text = [VOCAB.index2word(t) for t in continuation]
print(continuation_text)
print("Final:")
print(input_prompt + ' ' + ' '.join(continuation_text))


# # LSTM From Scratch (40 Points)

# Now we do LSTM the hard way---creating the LSTM cells by hand.
# 
# **Complete the following functions inside the `MyLSTMCell` class.**
# 
# We have broken the forward function into multiple parts:
# - Forget gate: determine what of the previous cell state should be discarded (by multiplying 0 or 1 produced by a sigmoid against the cell state).  `forget_gate()` implements $f=\sigma(W_{i,f}x+b_{i,f} + W_{h,f}h+b_{h,f})$.
# - Input gate: determine what of the input should be introduced to the cell memory. `input_gate()` implements $i=\sigma(W_{i,i}x+b_{i,i}+W_{h,i}h+b_{h,i})$
# - Cell memory: update the previous cell memory state to make a new cell memory state. `cell_memory()` implements $c'=f*c + i*tanh(W_{i,g}x+b_{i,g} + W_{h,g}h + b_{h,g})$.
# - Output gate: determine what from the current cell memory state. `output_gate()` implements $o=\sigma(W_{i,o}x+b_{i,o}+W_{h,o}h+b_{h,o})$.
# - A final function `hidden_out()` will produce the new hidden state by implementing $h'=o*tanh(c')$.
# 
# You will also need to initialize any linear layer modules, activation functions, etc. in the constructor.

# In[21]:


class MyLSTMCell(torch.nn.Module):

  def __init__(self, input_size=10, hidden_size=64):
    super(MyLSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    ### BEGIN SOLUTION
    # so first we need forget gate
    self.lin_forget_x_layer = nn.Linear(input_size, hidden_size)
    self.lin_forget_hid_layer = nn.Linear(hidden_size, hidden_size)
    # ok so that should be simple enough
    # now input gate
    self.lin_input_x_layer = nn.Linear(input_size, hidden_size)
    self.lin_input_hid_layer = nn.Linear(hidden_size, hidden_size)
    # now the cell memory
    self.lin_cellmem_x_layer = nn.Linear(input_size, hidden_size)
    self.lin_cellmem_hid_layer = nn.Linear(hidden_size, hidden_size)
    # and finally output
    self.lin_output_x_layer = nn.Linear(input_size, hidden_size)
    self.lin_output_hid_layer = nn.Linear(hidden_size, hidden_size)

    # kind of overkill but need these later
    self.sig = nn.Sigmoid()
    self.tan = nn.Tanh()
    ### END SOLUTION

  ### The Forget Gate takes in the input (x) and hidden state (h)
  ### The input and hidden state pass through their own linear compression layers,
  ### then are concatenated and passed through a sigmoid
  def forget_gate(self, x, h):
    f = None # The gate vector to return
    ### BEGIN SOLUTION
    # I think we just gotta go sigmoid here
    # linear transform first
    lin_trans = self.lin_forget_x_layer(x) + self.lin_forget_hid_layer(h)
    f = self.sig(lin_trans)
    ### END SOLUTION
    return f

  ### The Input Gate takes the input (x) and hidden state (h)
  ### The input and hidden state pass through their own linear compression layers,
  ### then are concatenated and passed through a sigmoid
  def input_gate(self, x, h):
    i = None # The gate vector to return
    ### BEGIN SOLUTION
    # same as forget here but just with input
    lin_trans = self.lin_input_x_layer(x) + self.lin_input_hid_layer(h)
    i = self.sig(lin_trans)
    ### END SOLUTION
    return i

  ### The Cell memory gate takes the results from the input gate (i), the results from the forget gate (f)
  ### the original input (x), the hidden state(h) and the previous cell state (c_prev).
  ### 1. The Cell memory gate compresses the input and hidden and concatenates them and passes it through a Tanh.
  ### 2. The resultant intermediate tensor is multiplied by the results from the input gate to determine
  ###    what new information is allowed to carry on
  ### 3. The results from the forget state are multiplied against the previous cell state (c_prev) to determine
  ###    what should be removed from the cell state.
  ### 4. The new cell state (c_next) is the new information that survived the input gate and the previous
  ###    cell state that survived the forget gate.
  ### The new cell state c_next is returned
  def cell_memory(self, i, f, x, h, c_prev):
    c_next = None
    ### BEGIN SOLUTION
    # literally just following the instructions in the above
    lin_trans = self.lin_cellmem_x_layer(x) + self.lin_cellmem_hid_layer(h)
    # tanh this time from the instructions
    cm = self.tan(lin_trans)
    # where the magic happens, new cell state
    c_next = f * c_prev + i * cm
    ### END SOLUTION
    return c_next

  ### The Out gate takes the original input (x) and the hidden state (h)
  ### The gate passes the input and hidden through their own compression layers and
  ### then concatenates to send through a sigmoid
  def out_gate(self, x, h):
    o = None # The gate vector to return
    ### BEGIN SOLUTION
    # back to simple like input and forget
    lin_trans = self.lin_output_x_layer(x) + self.lin_output_hid_layer(h)
    # back to sigmoid
    o = self.sig(lin_trans)
    ### END SOLUTION
    return o

  ### This function assembles the new hidden state, give the results of the output gate (o)
  ### and the new cells sate (c_next).
  ### This function runs c_next through a tanh to get a 1 or -1 which will flip some of the
  ### elements of the output.
  def hidden_out(self, o, c_next):
    h_next = None
    ### BEGIN SOLUTION
    # just using the output gate here
    # tanh instead of sigmoid here as described
    h_next = o * self.tan(c_next)
    ### END SOLUTION
    return h_next

  def forward(self, x, hc):
    (h, c_prev) = hc
    # Equation 1. input gate
    i = self.input_gate(x, h)

    # Equation 2. forget gate
    f = self.forget_gate(x, h)

    # Equation 3. updating the cell memory
    c_next = self.cell_memory(i, f, x, h, c_prev)

    # Equation 4. calculate the main output gate
    o = self.out_gate(x, h)

    # Equation 5. produce next hidden output
    h_next = self.hidden_out(o, c_next)

    return h_next, c_next

  def init_hidden(self):
    return (torch.zeros(1, self.hidden_size),
            torch.zeros(1, self.hidden_size))


# In[22]:


# student check - the following test must return a value of 6 to receive credit (5 pts)
ag.test_myLSTMCell_structure(MyLSTMCell)


# In[23]:


# student check - the following test must return a value of 8 to receive credit (5 pts)
ag.MyLSTMCell_linear_layer_size_check()


# Let's build a cell. A cell doesn't do much by itself.

# In[24]:


cell = MyLSTMCell(input_size=VOCAB.num_words(), hidden_size=64)


# In[25]:


# student check - the following test must return a value of 22 to receive credit (10 pts)
ag.test_gate_structure(cell)


# Now let's load your `MyLSTMCell` class into `MyLSTM`.

# In[ ]:


# It's ok to change this cell, however, you should not need to change it much (if at all) - note: certain changes may break the autograder, e.g., 
# increasing the size of the hidden layer could cause out of memory errors in the autograder and large numbers of epochs could cause autograder to time
# out (pay attention to the runtime of your notebook and the warnings that are printed out at the end of the notebook)
MY_CELL_HIDDEN_SIZE = 32
MY_CELL_NUM_EPOCHS = 3
MY_CELL_LEARNING_RATE = 0.01


# In[27]:


my_cell_lstm = MyLSTM(input_size=VOCAB.num_words(), hidden_size=MY_CELL_HIDDEN_SIZE, cell_type=MyLSTMCell)
optimizer_my_cell = optim.SGD(my_cell_lstm.parameters(), lr=MY_CELL_LEARNING_RATE)
criterion_my_cell = nn.NLLLoss()


# ## LSTM From Scratch---Training

# Lets see if your combination of `MyLSTM` using `MyLSTMCell` learns. We don't need to update the training loop

# In[28]:


epoch_losses = train_lstm(my_cell_lstm, optimizer_my_cell, criterion_my_cell, num_epochs=MY_CELL_NUM_EPOCHS, data=TRAIN)


# In[29]:


plt.figure(1)
plt.clf()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(len(epoch_losses)))
plt.plot(epoch_losses)


# You should see a curve that slopes down steeply at first and then levels out to some asymptotic minimum.

# ## LSTM From Scratch---Testing

# We don't need to update the evaluation loop

# In[30]:


# student check - the following test must return a value less than 1000 to receive credit (20 pts)
ag.eval_lstm_2(max_perplexity=1000)


# ## LSTM From Scratch---Generation

# Generation works the same.

# In[31]:


# It's ok to change this cell
MY_CELL_TEMPERATURE = 0.5


# In[32]:


# Example input prompt:
input_prompt = "the First War began"
# How long should the continuation be?
num_new_tokens = 10

# Normalize the input
normalized_input = normalize_string(input_prompt)
# Tokenize the input
tokenized_input = [VOCAB.word2index(w) for w in normalized_input.split()]
print("input prompt:", input_prompt)
print("input tokens:", tokenized_input, '\n')

# Get the hidden state that represents the input prompt
print("Prepping hidden state:\n")
hidden_state = prep_hidden_state(tokenized_input, my_cell_lstm, verbose=True)

# Generate the continuation. Use the argmax function to sample from the RNN's outputs
token = tokenized_input[-1]
continuation = generate_rnn(my_cell_lstm, num_new_tokens, token, hidden_state, fn=lambda d:my_temperature_sample(d, MY_CELL_TEMPERATURE), verbose=True)

# All done
print("Final continuation:")
print(continuation)
continuation_text = [VOCAB.index2word(t) for t in continuation]
print(continuation_text)
print("Final:")
print(input_prompt + ' ' + ' '.join(continuation_text))


# # Attention (40 Points)
# 
# Attention allows the network to look back at previous data when trying to predict the next token.
# 
# We will split the LSTM into an Encoder and a Decoder. The Encoder's job will be to update the hidden state based on the latest token. The Decoder's job is to predict the next token (log softmax over the vocabulary) based on the current hidden state as well as *n* previous hidden states. You will see that the training loop now collects up a stack of hidden states to pass to the Decoder. The Decoder will figure out how much the network should attend to each of the *n* prior hidden states before making its final prediction.
# 
# The Encoder will be a simple `nn.LSTMCell`. While the encoder could be more complicated, this allows us to focus on the Decoder. The Decoder is more complicated, involving both an LSTMCell and an attention mechanism.
# 
# **Complete the class defnition below**
# 
# `MyAttentionDecoder` will implement another `nn.LSTMCell` plus an attention mechanism.
# 
# Inputs:
# - `x`: a one-hot of the current token as a `1 x vocab_size` tensor
# - `hc`: a tuple containing a tuple with encoder's hidden state and memory cell state. The hidden state and cell state are both `1 x hidden_size` tensors.
# - `encoder_outputs`: a history of *n* encoded hidden states, as a `n x hidden_size` tensor (this data is not batched).
# 
# Outputs:
# - `h_hat`: a log softmax probability distribution over the vocabular, as a `1 x vocab_size` tensor
# - `hc_out`: a tuple containing the LSTMCell's hidden state and memory cell state. The hidden state and cell state are both `1 x hidden_size` tensors.

# In[33]:


class MyAttentionDecoder(nn.Module):
  def __init__(self, hidden_size, input_size, max_length):
    super(MyAttentionDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.max_length = max_length
    ### BEGIN SOLUTION
    # adding this infront of the LSTM cell hwihc I THINK I need
    self.embedding = nn.Linear(input_size, hidden_size)
    # grab our LSTM cell we just created
    # NOW LSTM cell
    self.lstm_cell_lin = nn.LSTMCell(hidden_size, hidden_size)
    # now the attention
    # i think i need this to concatenate the hidden and embedded layers
    hidsize = hidden_size*2
    self.attn_layer1 = nn.Linear(hidsize, max_length)
    # attention combine
    # still think I need to double hidden size here, well see
    self.attn_combine1 = nn.Linear(hidsize, hidden_size)
    # output layer
    # normal linear guy here
    self.out1 = nn.Linear(hidden_size, input_size)
    # i keep getting yelled at by the test that I need relu
    self.relu = nn.ReLU()
    # softmax
    self.logsoft = nn.LogSoftmax(dim=1)

    # think that should do it

    ### END SOLUTION

  def forward(self, x, hc, encoder_outputs):
    log_probs = None
    hc_out = None
    ### BEGIN SOLUTION
    # tedious part, connecting it all back
    # break this up
    hid, cstate = hc

    # Embed the input to reduce dimensionality
    # setting up the embedding as above that I think I need
    embed1 = self.embedding(x)
    embed1 = self.relu(embed1)

    # concat and get weights
    tup = (embed1, hid)
    attn_in_it = torch.cat(tup, 1)
    atn_wgts = F.softmax(self.attn_layer1(attn_in_it), dim=1)

    # applied attn to encoder outputs
    atn_apld = torch.bmm(atn_wgts.unsqueeze(0), encoder_outputs.unsqueeze(0))

    # combuine embeds and attntion context now
    comb_tup = (embed1, atn_apld[0])
    comb = torch.cat(comb_tup, 1)
    comb = self.attn_combine1(comb)
    comb = self.relu(comb)

    h_c_tup = (hid, cstate)
    h_n, c_n = self.lstm_cell_lin(comb, h_c_tup)

    # output probs
    out = self.out1(h_n)
    log_ouput_final = self.logsoft(out)
    # crap, var names
    log_probs = log_ouput_final
    hc_out = (h_n, c_n)

    ### END SOLUTION
    return log_probs, hc_out

  def init_hidden(self):
    return (torch.zeros(1, self.hidden_size),
            torch.zeros(1, self.hidden_size))


# In[ ]:


# It's ok to change this cell, however, you should not need to change it much (if at all) - note: certain changes may break the autograder, e.g., 
# increasing the size of the hidden layer could cause out of memory errors in the autograder and large numbers of epochs could cause autograder to time
# out (pay attention to the runtime of your notebook and the warnings that are printed out at the end of the notebook)
ATTN_MAX_LENGTH = 5  # The number of past hidden states that can be attended to
ATTN_HIDDEN_SIZE = 32
ATTN_NUM_EPOCHS = 3
ATTN_LEARNING_RATE = 0.01


# In[35]:


attn_encoder = nn.LSTMCell(VOCAB.num_words(), ATTN_HIDDEN_SIZE)
attn_decoder = MyAttentionDecoder(ATTN_HIDDEN_SIZE, VOCAB.num_words(), ATTN_MAX_LENGTH)
attn_criterion = nn.NLLLoss()
attn_encoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=ATTN_LEARNING_RATE, momentum=0.9)
attn_decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=ATTN_LEARNING_RATE, momentum=0.9)


# In[36]:


# student check - the following test must return a value of 4 to receive credit (10 pts)
ag.test_attention_structure()


# In[37]:


# student check - the following test must return a value of 8 to receive credit (10 pts)
ag.attention_linear_layer_size_check()


# ## Attention---Training

# The training loop is a bit more involved because it must collect up a number of past hidden states. It still uses your `get_rnn_x_y()` function though.

# In[38]:


def train_attn(data, num_epochs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=ATTN_MAX_LENGTH):
  epoch_losses = []
  encoder.train()
  decoder.train()
  for epoch in range(num_epochs):
    losses = []
    # Get an empty hc
    encoder_hc = decoder.init_hidden()
    # Create an empty history of hiddens
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    for iter in range(len(data)-1):
      x, y = ag.get_rnn_x_y(data, iter, VOCAB.num_words())
      x = x.float()
      # Call encoder
      encoder_hidden, encoder_cell = encoder(x, encoder_hc)
      encoder_hc = (encoder_hidden, encoder_cell)
      # unbatch the hidden so it can be added to encoder_outputs
      encoder_output = encoder_hidden[0]
      # Shift all the previous outputs
      # Grab elements 1...max (dropping row 0) and flatten
      encoder_outputs = encoder_outputs[1:,:].view(-1)
      # Add the new output
      encoder_outputs = torch.cat((encoder_outputs, encoder_output.detach()))
      # re-fold
      encoder_outputs = encoder_outputs.view(max_length, -1)
      # decoder's input hc is the encoder's output hc
      decoder_hc = encoder_hc
      # Call the decoder
      decoder_output, decoder_hc = decoder(x, decoder_hc, encoder_outputs)

      loss = criterion(decoder_output, y)
      losses.append(loss)

      encoder_optimizer.zero_grad()
      decoder_optimizer.zero_grad()
      loss.backward()
      encoder_optimizer.step()
      decoder_optimizer.step()

      # Prep the decoder hc for the next iteration
      encoder_hc = (decoder_hc[0].detach(), decoder_hc[1].detach())

      if iter%1000 == 0:
        print("iter", iter, "loss", torch.stack(losses).mean().item())
    print("epoch", epoch, "loss", torch.stack(losses).mean().item())
    epoch_losses.append(torch.stack(losses).mean().item())
  return epoch_losses


# In[39]:


epoch_losses = train_attn(TRAIN, ATTN_NUM_EPOCHS, attn_encoder, attn_decoder, attn_encoder_optimizer, attn_decoder_optimizer, attn_criterion, ATTN_MAX_LENGTH)


# In[40]:


plt.figure(1)
plt.clf()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(len(epoch_losses)))
plt.plot(epoch_losses)


# You should see a curve that slopes down steeply at first and then levels out to some asymptotic minimum.

# ## Attention---Testing

# In[41]:


# student check - the following test must return a value less than 1000 to receive credit (20 pts)
ag.eval_attn(max_perplexity=1000)


# # Grading
# 
# Please submit this .ipynb file to Gradescope for grading.

# ## Final Grade

# In[42]:


# student check
ag.final_grade()


# ## Notebook Runtime

# In[43]:


# end time - notebook execution
end_nb = time.time()
# print notebook execution time in minutes
print("Notebook execution time in minutes =", (end_nb - start_nb)/60)
# warn student if notebook execution time is greater than 30 minutes
if (end_nb - start_nb)/60 > 30:
  print("WARNING: Notebook execution time is greater than 30 minutes. Your submission may not complete auto-grading on Gradescope. Please optimize your code to reduce the notebook execution time.")

