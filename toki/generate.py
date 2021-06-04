from argparse import ArgumentParser
import sys
import random as rd
from .generator import TokiGenerator, ProperTransformer, SoweliConverter
from tunimi.wannimi import Detokenizer

def main():
    parser = ArgumentParser()
    parser.add_argument('-n', type = int, default = 1)
    parser.add_argument('--checkpoint', default='checkpoint.pt')
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=6)
    args = parser.parse_args()

    generator = TokiGenerator(args.checkpoint, args.hidden_size, args.nhead, args.num_layers)
    pt = ProperTransformer()
    sc = SoweliConverter()
    detokenizer = Detokenizer()

    for _ in range(args.n):
        x = generator(as_str = True)
        x = pt(x)
        if rd.random() < 0.5:
            x = sc(x)
        x = detokenizer(x)
        print(x)

