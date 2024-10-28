import pdb

# Just run this block. Please do not modify the following code.
import math
import time
import io
import numpy as np
import csv
from IPython.display import Image

# Pytorch package
import torch
import torch.nn as nn
import torch.optim as optim

# Torchtest package
import torchtext
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
from torch.nn.utils.rnn import pad_sequence

# Tqdm progress bar
from tqdm import tqdm_notebook, tqdm

from utils import set_seed_nb, unit_test_values

import seq2seq.Encoder
import seq2seq.Decoder

# @title
from Transformer import FullTransformerTranslator

# you will be implementing and testing the forward function here. During training, inaddition to inputs, targets are also passed to the forward function
set_seed_nb()
train_inxs = np.load('../data/train_inxs.npy')

# load dictionary
word_to_ix = {}
with open("../data/word_to_ix.csv", "r", encoding='utf-8') as f:
    reader = csv.reader(f)
    for line in reader:
        word_to_ix[line[0]] = line[1]
# print("Vocabulary Size:", len(word_to_ix))

# inputs = train_inxs[0:3]
# inputs[:,0]=0
# inputs = torch.LongTensor(inputs)
# inputs.to('cpu')
# # Model
# full_trans_model = FullTransformerTranslator(input_size=len(word_to_ix), output_size=5, device='cpu', hidden_dim=128, num_heads=2, dim_feedforward=2048, max_length=train_inxs.shape[1]).to('cpu')

# tgt_array = np.random.rand(inputs.shape[0], inputs.shape[1])
# targets = torch.LongTensor(tgt_array)
# targets.to('cpu')
# outputs = full_trans_model.forward(inputs,targets)

# if outputs is not None:
#     expected_out = unit_test_values('full_trans_fwd')
#     print('Close to outputs: ', expected_out.allclose(outputs, atol=1e-4))
#     # print('Outputs:', outputs)
#     # print('Expected Outputs:', expected_out)
# else:
#     print("NOT IMPLEMENTED")


# you will be implementing the generate_translation function which is called at the time of interence to translate the inputs. This is done in an autoregressive manner very similar to how you implemented the seq2seq model earlier
inputs = train_inxs[3:6]
inputs[:,0]=0
inputs = torch.LongTensor(inputs)
inputs.to('cpu')
full_trans_model = FullTransformerTranslator(input_size=len(word_to_ix), output_size=5, device='cpu', hidden_dim=128, num_heads=2, dim_feedforward=2048, max_length=train_inxs.shape[1]).to('cpu')
expected_out = unit_test_values('full_trans_translate')
print(expected_out[0])
outputs = full_trans_model.generate_translation(inputs)


if outputs is not None:
    print(expected_out[0])
    print(outputs[0])
    print('Close to outputs: ', expected_out.allclose(outputs, atol=1e-4))
else:
    print("NOT IMPLEMENTED")