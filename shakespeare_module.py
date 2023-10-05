import re
import torch 
from torch.utils.data import DataLoader
import requests
from fastai.text.all import *

import torchtext.transforms as T
import spacy
from torchtext.vocab import build_vocab_from_iterator

class ShakespeareDataSet(torch.utils.data.Dataset):
    def __init__(
            self, split: 'val' or 'train', seq_len: int=16,
            batch_size: int = 64, vocab_size: int=30_000
            ):
        super(ShakespeareDataSet, self).__init__()
        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.eng = spacy.load("en_core_web_sm") # Load the English model to tokenize English text
        self.prepare_data()
        self.configure_dataloader()

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index, :]
        y = self._y[index, :]
        return x, y
    
    def engTokenize(self, text):
        """
        Tokenize an English text and return a list of tokens
        """
        return [token.text for token in self.eng.tokenizer(text)]

    def get_data(self):
        link = "https://www.gutenberg.org/files/100/100-0.txt"
        f = requests.get(link)
        lines = f.text
        return lines.split(
        "VENUS AND ADONIS", 1)[1].split(
        "*** END OF THE PROJECT GUTENBERG", 1)[0][:100_000]

    def prepare_data(self):
        lines = self.get_data()
        spacy = WordTokenizer()
        tkn = Tokenizer(spacy)
        toks = tkn(lines)
        num = Numericalize(max_vocab=self.vocab_size)
        num.setup(toks)
        nums = num(toks)
        X = torch.Tensor()
        y = torch.Tensor()
        for i in range(0, len(nums)-self.seq_len-1, self.seq_len):
            X = torch.cat((X, nums[i:i+self.seq_len]), 0)
            y = torch.cat((y, nums[i+1:i+self.seq_len+1]), 0)
        print(X.shape)
        print(y.shape)
        cut = int(len(X) * 0.8)
        if self.split == 'train':
            self._x = torch.tensor(X[:cut], dtype=torch.float32)
            self._y = torch.tensor(y[:cut], dtype=torch.float32)
        elif self.split == 'val':
            self._x = torch.tensor(X[cut:], dtype=torch.float32)
            self._y = torch.tensor(y[cut:], dtype=torch.float32)


    def configure_dataloader(self):
        self.dataloader = DataLoader(self, batch_size=self.batch_size, shuffle=True)