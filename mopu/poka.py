import torch
from argparse import ArgumentParser
from tunimi.vocabulary import Vocabulary
from soweli.toki import SoweliToki
from .sampler import SentenceSampler
from .util import load_vocab_and_model
from .proper import ProperTransformer
from .extractor import ProperExtractor
from .jansoweli import JanSoweliConverter
from .punctnormalizer import PunctNormalizer
from tunimi.normalizer import Normalizer
from tunimi.tokenizer import Tokenizer
from tunimi.wannimi import Detokenizer

def main():
    parser = ArgumentParser()
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
    js_converter = JanSoweliConverter(vocab)
    p_transformer = ProperTransformer()
    p_extractor = ProperExtractor()
    punctnorm = PunctNormalizer()
    normalizer = Normalizer()
    tokenizer = Tokenizer(vocab = vocab)
    detokenizer = Detokenizer()

    while True:
        utt = input()
        utt = punctnorm(utt)
        sent = '"{}" "'.format(utt)
        sent = normalizer(sent)
        utt_proper_list = p_extractor(sent)
        sent = tokenizer(sent)
        len_utt = len(sent)
        sent = js_converter(sent)
        sent = sampler(sent, stop_by_quot = True)
        sent = sent[len_utt - 1 : ]
        sent = js_converter(sent)
        sent = [vocab[word] for word in sent]
        sent = p_transformer(sent, name_list = utt_proper_list)
        sent = detokenizer(sent)
        sent = sent.strip('"')
        print(sent)

