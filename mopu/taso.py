import random as rd
import torch
from argparse import ArgumentParser
from tunimi.vocabulary import Vocabulary
from soweli.toki import SoweliToki
from .sampler import SentenceSampler
from .util import load_vocab_and_model
from .proper import ProperTransformer
from .jansoweli import JanSoweliConverter
from tunimi.wannimi import Detokenizer

def main():
    parser = ArgumentParser()
    parser.add_argument('--iter', default = 10, type = int)
    parser.add_argument('--hidden-size', default = 512, type = int)
    parser.add_argument('--nhead', default = 8, type = int)
    parser.add_argument('--num-layers', default = 6, type = int)
    parser.add_argument('--checkpoint-path', default = 'checkpoint.pt')
    args = parser.parse_args()

    hidden_size = args.hidden_size
    nhead = args.nhead
    num_layers = args.num_layers
    checkpoint_path = args.checkpoint_path

    vocab, model = load_vocab_and_model(
            hidden_size, nhead, num_layers, checkpoint_path)
    sampler = SentenceSampler(vocab, model)
    p_transformer = ProperTransformer()
    js_converter = JanSoweliConverter(vocab)
    detokenizer = Detokenizer()
    soweli_threshold = 0.5

    for _ in range(args.iter):
        sent = sampler()
        if rd.random() < soweli_threshold:
            sent = js_converter(sent)
        sent = [vocab[word] for word in sent]
        sent = p_transformer(sent)
        sent = detokenizer(sent)
        print(sent)

