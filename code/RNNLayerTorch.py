import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor

class RNNLayerTorch(nn.Module):
    """ Custom Layer for reccurent network """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        w_hx = Tensor(size_out, size_in)
        self.w_hx = nn.Parameter(w_hx)
        bias = Tensor(size_out)
        self.bias = nn.Parameter(bias)
        w_hh = Tensor(size_out, size_out)
        self.w_hh = nn.Parameter(w_hh)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.w_hh, a=math.sqrt(5)) # weight init
        nn.init.kaiming_uniform_(self.w_hx, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_hx)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x, hidden_state):
        x_w = F.linear(x, self.w_hx, bias=self.bias)
        h_w = F.linear(hidden_state, self.w_hh)
        return F.relu(x_w + h_w)
