import torch
from torch.nn.utils.rnn import pad_sequence as pad
from .batch import TokiBatch
from pali.dataset import Dataset
from soweli.util import generate_square_subsequent_mask

class TokiDataset(Dataset):
    def collate(self, batch):
        i = pad([torch.tensor([self.eos] + sent) for sent in batch], padding_value = self.pad)
        o = pad([torch.tensor(sent + [self.eos]) for sent in batch], padding_value = self.pad)
        l = torch.tensor([len(sent) + 1 for sent in batch])
        am = generate_square_subsequent_mask(i.shape[0])
        pm = (i == self.pad).T
        return TokiBatch(i, o, l, am, pm)

