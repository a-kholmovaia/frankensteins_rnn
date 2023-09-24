import torch
import torch.nn as nn
import torch.nn.functional as F

class TheSonnetsRNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super(TheSonnetsRNN, self).__init__()
        self.hidden_size = hidden_size
        input_size = data_size + hidden_size

        self.i2h1 = nn.Linear(input_size, hidden_size)
        self.i2h1 = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output