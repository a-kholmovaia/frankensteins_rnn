import torch
import torch.nn as nn
from RNNLayerTorch import RNNLayerTorch

class FrankensteinsRNN(nn.Module):
    def __init__(self, n_units_h1, n_units_h2,
                 vocab_size, seq_len):
        super(FrankensteinsRNN, self).__init__()
        self.n_units_h1 = n_units_h1
        self.n_units_h2 = n_units_h2
        self.seq_len = seq_len
        self.h1 = torch.zeros(1, n_units_h1).cuda()
        self.h2 = torch.zeros(1, n_units_h2).cuda()
        self.embedding = nn.Embedding(vocab_size, n_units_h1)
        self.ih1 = RNNLayerTorch(n_units_h1, n_units_h1)
        self.h1h2 = RNNLayerTorch(n_units_h1, n_units_h2)
        self.h2o = nn.Linear(n_units_h2, vocab_size)

    def reset_hidden_state(self):
        self.h1.zero_()
        self.h2.zero_()

    def forward(self, x):
        outs = []
        x = x[0]
        for i in range(self.seq_len):
            x_ = self.embedding(x[i])
            self.h1 = self.ih1(x_, self.h1)
            self.h2 = self.h1h2(self.h1, self.h2)
            outs.append(self.h2o(self.h2))
        self.h1 = self.h1.detach()
        self.h2 = self.h2.detach()
        return torch.stack(outs, dim=1)
