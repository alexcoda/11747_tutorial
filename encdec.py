import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
from utils import use_cuda


#todo: rnn constructor taking rnn_type (GRU vs LSTM), num_layers, etc as args
#todo: minibatching

class RNNEncoder(nn.Module):
    """simple initial encoder"""
    def __init__(self, input_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.input_size  = input_size  #source vocab size
        self.hidden_size = hidden_size
        self.embeddings  = nn.Embedding(input_size, hidden_size) #embedding dim = hidden_size
        self.rnn = nn.GRU(hidden_size, hidden_size)

    #src is currently single sentence
    def forward(self, src, hidden):
        embedded = self.embeddings(src).view(1, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


class RNNDecoder(nn.Module):
    """simple initial decoder"""
    def __init__(self, output_size, hidden_size):
        super(RNNDecoder, self).__init__()
        self.output_size = output_size  #target vocab size
        self.hidden_size = hidden_size
        self.embedding   = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


class EncDec(nn.Module):
    """initial encoder + decoder model"""
    def __init__(self, encoder, decoder):
        super(EncDec, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    #src,tgt currently single sentences
    def forward(self, src, tgt, lengths, dec_state=None):
        encoder_outputs, encoder_hidden = self.encoder(src)
        output = self.decoder(tgt, encoder_hidden, encoder_outputs)
        return decoder_outputs

    def save(self, fname):
        """Save the model to a pickle file."""
        with open(fname, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
