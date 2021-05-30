import random as rd
import torch
from torch.nn.utils.rnn import pad_sequence as pad
from .batch import Batch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sents, vocab):
        self.sents = sents
        self.vocab = vocab
        self.pad = self.vocab.pad_id
        self.eos = self.vocab.eos_id

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        return self.sents[index]

