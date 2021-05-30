from pali.batch import Batch

class TokiBatch(Batch):
    def __init__(self, i, o = None, l = None, am = None, pm = None):
        self.inputs = i
        self.outputs = o
        self.lengths = l
        self.attention_mask = am
        self.padding_mask = pm

    def __len__(self):
        return self.inputs.shape[1]

    def get_lengths(self):
        return self.lengths

    def cuda(self):
        self.inputs = self.inputs.cuda()

        if self.outputs is not None:
            self.outputs = self.outputs.cuda()

        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.cuda()

        if self.padding_mask is not None:
            self.padding_mask = self.padding_mask.cuda()

        return self

