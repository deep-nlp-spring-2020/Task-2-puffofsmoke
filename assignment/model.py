#  Copyright (c) 2020
#  roman.grigorov@gmail.com aka PuffOfSmoke or dePuff
import torch
from torch import nn as nn


class RNNLM(nn.Module):
    def __init__(self, ntoken, emsize, nhid, padding_idx, nlayers=1, dropout_probability=0, mode='gru'):
        super().__init__()

        self.nlayers = nlayers
        self.dropout_probability = dropout_probability

        self.ntoken = ntoken
        self.emsize = emsize
        self.nhid = nhid

        self.encoder = nn.Embedding(ntoken, emsize, padding_idx=padding_idx)

        if mode == 'lstm':
            self.rnn = nn.LSTM(emsize, nhid, num_layers=nlayers, batch_first=True, dropout=dropout_probability)
        else:
            self.rnn = nn.GRU(emsize, nhid, num_layers=nlayers, batch_first=True, dropout=dropout_probability)

        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

    def init_weights(self):
        range_init = 0.1
        self.encoder.weight.data.uniform_(-range_init, range_init)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-range_init, range_init)

    def forward(self, inputs, hidden=None):
        encoded = self.encoder(inputs)

        if hidden is None:
            output, hidden = self.rnn(encoded)
        else:
            output, hidden = self.rnn(encoded, hidden)
        decoded = self.decoder(output).view(-1, self.ntoken)

        # Loss changed to CrossEntropyLoss no need softmax here
        # sm = F.log_softmax(decoded, dim=1)

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNNBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim, vocab_size, padding_idx=1, mode='gru'):
        super().__init__()

        self.pooling = None

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)

        if mode == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim)
        else:
            self.rnn = nn.GRU(emb_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, 1)

        # self.init_weights()

    # def init_weights(self):
    #     range_init = 0.1
    #     self.emb.weight.data.uniform_(-range_init, range_init)
    #     self.out.bias.data.zero_()
    #     self.out.weight.data.uniform_(-range_init, range_init)

    def forward(self, seq):
        embs = self.emb(seq)

        self.rnn.flatten_parameters()
        rnn_out, hidden = self.rnn(embs)

        if self.pooling == 'max':
            hidden = torch.max(hidden, dim=0).values
        elif self.pooling == 'average':
            hidden = torch.sum(hidden, dim=0) / hidden.size(0)
        else:
            hidden = hidden[-1]

        # ToDo remove this compatibility hack
        return self.out(hidden).squeeze(), True

    # def init_hidden(self, batch_size):
    #     return Variable(torch.zeros((1, batch_size, self.hidden_dim))).cuda()


class RNNAdvanced(RNNBaseline):
    def __init__(self, hidden_dim, emb_dim, vocab_size, padding_idx=1, mode='gru', bidirectional=False, pooling=None):
        super().__init__(hidden_dim, emb_dim, vocab_size, padding_idx, mode)
        if mode == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(emb_dim, hidden_dim, bidirectional=bidirectional)
        self.pooling = pooling


class CNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, kernel_sizes, dropout=0.5, num_filters=5):
        super().__init__()

        self.num_filters = num_filters
        self.dropout = dropout

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=1)

        self.cnns = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=emb_dim,
                                    out_channels=self.num_filters,
                                    kernel_size=kernel_size),
                          # nn.ReLU(),
                          nn.AdaptiveMaxPool1d(1)
                          )
            for kernel_size in kernel_sizes
        ])
        self.decoder = nn.Linear(self.num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        embs = self.emb(x).permute(0, 2, 1)
        encoded = [cnn(embs) for cnn in self.cnns]
        encoded = torch.cat(encoded, dim=1).flatten(start_dim=1)  # [bs, num_filters*len(kernels)]
        logit = self.decoder(encoded).squeeze()
        # ToDo remove this compatibility hack (, True)
        return logit, True
