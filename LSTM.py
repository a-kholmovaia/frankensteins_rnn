import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, embedding_dim: int, 
                 num_layers: int,
                 n_units: int, 
                 vocab_size: int, 
                 seq_len: int):
        super(LSTM, self).__init__()
        self.lstm_size = n_units
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.st1, self.st2 = self.init_state(seq_len)
        self.fc = nn.Linear(self.lstm_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        output, (self.st1, self.st2) = self.lstm(embed, (self.st1, self.st2))
        logits = self.fc(output)

        self.st1 = self.st1.detach()
        self.st2 = self.st2.detach()
        return logits

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size).cuda(),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size).cuda())