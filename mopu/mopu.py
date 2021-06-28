import re
import sys
import time
import random as rd
import tweepy
import torch
from argparse import ArgumentParser
from tunimi.vocabulary import Vocabulary
from .sampler import SentenceSampler
from .util import load_vocab_and_model
from .proper import ProperTransformer
from .extractor import ProperExtractor
from .jansoweli import JanSoweliConverter
from .punctnormalizer import PunctNormalizer
from tunimi.normalizer import Normalizer
from tunimi.tokenizer import Tokenizer
from tunimi.wannimi import Detokenizer

class Mopu:
    def __init__(self, hidden_size, nhead, num_layers, checkpoint_path):
        vocab, model = load_vocab_and_model(
                hidden_size, nhead, num_layers, checkpoint_path)
        self.vocab = vocab
        self.sampler = SentenceSampler(vocab, model)

        self.js_converter = JanSoweliConverter(vocab)
        self.p_transformer = ProperTransformer()
        self.p_extractor = ProperExtractor()
        self.punctnorm = PunctNormalizer()
        self.normalizer = Normalizer()
        self.tokenizer = Tokenizer(vocab = vocab)
        self.detokenizer = Detokenizer()

    def tweet(self):
        sent = self.sampler()
        if rd.random() < self.soweli_threshold:
            sent = self.js_converter(sent)
        sent = [self.vocab[word] for word in sent]
        sent = self.p_transformer(sent)
        sent = self.detokenizer(sent)
        return sent

    def reply(self, utt):
        utt = self.punctnorm(utt)
        sent = '"{}" "'.format(utt)
        sent = self.normalizer(sent)
        utt_proper_list = self.p_extractor(sent)
        sent = self.tokenizer(sent)
        len_utt = len(sent)
        sent = self.js_converter(sent)
        sent = self.sampler(sent, stop_by_quot = True)
        sent = sent[len_utt - 1 : ]
        sent = self.js_converter(sent)
        sent = [self.vocab[word] for word in sent]
        sent = self.p_transformer(sent, name_list = utt_proper_list)
        sent = self.detokenizer(sent)
        sent = sent.strip('"')
        return sent

class TimeManager:
    def __init__(self, N = 20):
        self.N = N
        self.reset()
        self.tweet = False
        self.reply = False

    def reset(self):
        gm = time.gmtime()
        self.m = gm.tm_min
        self.s = gm.tm_sec

    def update(self):
        gm = time.gmtime()
        if (self.s // self.N) != (gm.tm_sec // self.N):
            self.reset()
            self.tweet = True
            self.reply = True

    def tweetable(self):
        return self.tweet and self.m % 15 == 0 and (self.s // self.N) == 0

    def replyable(self):
        return self.reply

    def __str__(self):
        x = 'tweet: {}, m: {}, s: {}'.format(self.tweet, self.m, self.s)
        return x

def gen_api(CK, CS, AK, AS):
    auth = tweepy.OAuthHandler(CK, CS)
    auth.set_access_token(AK, AS)
    api = tweepy.API(auth)
    return api

def tweet(api, soweli, last):
    try:
        text = soweli.tweet()
        api.update_status(status = text)
        print(text)
    except Exception as e:
        print(e)
        print('failed')

def reply(api, soweli):
    try:
        mtl = api.mentions_timeline(since_id = last)
    except:
        mtl = []
    if mtl != []:
        last = mtl[0].id

    for m in mtl:
        try:
            utt = re.sub(r'@[^ ]+ ', '', m.text)
            name = m.user.screen_name
            stid = m.id
            text = soweli.reply(utt)
            status = '@{} {}'.format(name, text)
            api.update_status(status = status, in_reply_to_status_id = stid)
        except Exception as e:
            print(e)
            print('failed')
    return last

def main():
    parser = ArgumentParser()
    parser.add_argument('--consumer-key')
    parser.add_argument('--consumer-secret')
    parser.add_argument('--api-key')
    parser.add_argument('--api-secret')
    parser.add_argument('--hidden-size', default = 512, type = int)
    parser.add_argument('--nhead', default = 8, type = int)
    parser.add_argument('--num-layers', default = 6, type = int)
    parser.add_argument('--checkpoint-path', default = 'checkpoint.pt')
    args = parser.parse_args()

    soweli = Mopu(
            hidden_size = args.hidden_size,
            nhead = args.nhead,
            num_layers = args.num_layers,
            checkpoint_path = args.checkpoint_path)
    api = gen_api(
            CK = args.consumer_key,
            CS = args.consumer_secret,
            AK = args.api_key,
            AS = args.api_secret)
    tm = TimeManager()

    mtl = api.mentions_timeline()
    last = mtl[0].id

    while True:
        tm.update()
        print(tm, file = sys.stderr)

        if tm.tweetable():
            tweet(api, soweli)
            tm.tweet = False
        if tm.replyable():
            last = reply(api, soweli, last)
            tm.reply = False
        time.sleep(1)

