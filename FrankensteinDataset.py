import re
import torch 
from torch.utils.data import DataLoader

import spacy
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
import torchdata.datapipes as dp

class FrankensteinDataSet():
    PAD_SIZE = 77
    def __init__(
            self, split: 'val' or 'train', batch_size=1
            ):
        super(FrankensteinDataSet, self).__init__()
        self.split = split
        self.batch_size = batch_size
        self.eng = spacy.load("en_core_web_sm") # Load the English model to tokenize English text
        self.set_up()
        self.dl = self.get_dataloader()
        self.len = len(list(self.data_pipe))
    
    def engTokenize(self, text):
        """
        Tokenize an English text and return a list of tokens
        """
        return [token.text for token in self.eng.tokenizer(text)]

    def getTokens(self, data_iter):
        """
        Function to yield tokens from an iterator
        """
        for eng in data_iter:
            if len(eng) != 0:
                if re.match('[0-9 ]+', eng[0]) == None: # sort out the lines that only contains spaces and numbers
                    yield self.engTokenize(eng[0])
    
    def getTransform(self):
        """
        Create transforms based on given vocabulary. The returned transform is applied to sequence
        of tokens.
        """
        text_tranform = T.Sequential(
            ## converts the sentences to indices based on given vocabulary
            T.VocabTransform(vocab=self.source_vocab),
            ## Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
            # 1 as seen in previous section
            T.AddToken(1, begin=True),
            ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
            # 2 as seen in previous section
            T.AddToken(2, begin=False)
        )
        return text_tranform
    
    def applyTransform(self, sequence):
        """
        Apply transforms to sequence of tokens and create input & target vectors
        """
        tokenized = self.engTokenize(sequence[0])
        transformed = self.getTransform()(tokenized)

        return (transformed[:-1], # X
                transformed[1:]) # target
    
    def applyPadding(self, pair_of_sequences):
        """
        Convert sequences to tensors and apply padding
        """
        return (T.PadTransform(self.PAD_SIZE, 0)(T.ToTensor()(list(pair_of_sequences[0]))), 
            T.PadTransform(self.PAD_SIZE, 0)(T.ToTensor()(list(pair_of_sequences[1]))))
    
    def set_up(self):
        FILE_PATH = 'train.txt'
        data_pipe = dp.iter.IterableWrapper([FILE_PATH])
        data_pipe = dp.iter.FileOpener(data_pipe, mode='r')
        data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\r', as_tuple=False)
        self.source_vocab = build_vocab_from_iterator(
            self.getTokens(data_pipe),
            min_freq=2,
            specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
            special_first=True
        )
        self.source_vocab.set_default_index(self.source_vocab['<unk>']) # setting <unk> instead of that unknown word
        self.vocab_size = len(self.source_vocab)

    def preprocess_data(self):
        if self.split=='train':
            data_pipe = dp.iter.IterableWrapper(['train.txt'])
        else:
            data_pipe = dp.iter.IterableWrapper(['val.txt'])
        data_pipe = dp.iter.FileOpener(data_pipe, mode='r')
        data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='.', as_tuple=False)

        data_pipe = data_pipe.map(self.applyTransform) ## Apply the function to each element in the iterator
        data_pipe = data_pipe.map(self.applyPadding)
        return data_pipe

    def get_dataloader(self):
        data_pipe = self.preprocess_data()
        self.data_pipe = data_pipe
        return DataLoader(dataset=data_pipe, batch_size=self.batch_size)
    
   