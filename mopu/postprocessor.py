import random as rd
from tunimi.wannimi import Detokenizer
from .proper import ProperTransformer
from .jansoweli import JanSoweliConverter

class PostProcessor:
    def __init__(self, vocab, soweli_threshold = 0.5):
        self.vocab = vocab
        self.soweli_threshold = soweli_threshold
        self.js = JanSoweliConverter()
        self.pt = ProperTransformer()
        self.detokenizer = Detokenizer()

    def __call__(self, sent):
        sent = [self.vocab[word] for word in sent]
        if rd.random() < self.soweli_threshold:
            sent = self.js(sent)
        sent = self.pt(sent)
        sent = self.detokenizer(sent)
        return sent

