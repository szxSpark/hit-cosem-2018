#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
class biLSTM(nn.Module):
    def __init__(self, args, vocab_size, num_class):
        super(biLSTM, self).__init__()
        self.num_class = num_class
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=args.embedding_size)
        self.rnn = nn.LSTM(input_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           num_layers=args.num_layers,
                           bias=True,
                           batch_first=True,
                           dropout=args.dropout_rate,
                           bidirectional=True)
        self.fc = nn.Linear(args.hidden_size * 2, self.num_class)

    def forward(self, input):
        input = self.embedding(input)
        # input: batch_size x max_seq_len x embedding_dim
        output, (_, _) = self.rnn(input)
        # output: batch x max_seq_len x (hidden_size * num_directions * num_layers)

        logit = self.fc(output)
        # logit: batch x max_seq_len x num_class
        logit = logit.view(-1, self.num_class)
        # logit: (batch * max_seq_len) x num_class

        return logit

    def get_optimizer(self, lr, lr2, weight_decay):
        return torch.optim.Adam([
            {'params': self.embedding.parameters(), 'lr': lr2},
            {'params': self.rnn.parameters()},
            {'params': self.fc.parameters()}
        ], lr=lr, weight_decay=weight_decay)
