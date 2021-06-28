import torch
from .util import top_p_sampling, sent_to_inputs, process_logit

class SentenceSampler:
    def __init__(self, vocab, model,
            temperature = 1.0, top_p = 0.8,
            max_tokens = 64, stop_ratio = 0.5,
            no_specials = False, no_number = False, no_proper = False):
        self.vocab = vocab
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop_ratio = stop_ratio
        self.no_specials = no_specials
        self.no_number = no_number
        self.no_proper = no_proper

    def decode(self, x, am):
        with torch.no_grad():
            y = self.model.decode(x, attention_mask = am)
        logit = y[-1, 0, :]
        return logit

    def process_logit(self, i, logit):
        logit = process_logit(self.vocab, logit,
                no_specials = self.no_specials,
                no_number = self.no_number,
                no_proper = self.no_proper)
        if i > (self.max_tokens * self.stop_ratio):
            penalty = i - (self.max_tokens * self.stop_ratio) + 1
            logit[self.vocab.eos_id] = logit[self.vocab.eos_id] * penalty
        return logit

    def sample_next_token(self, logit):
        next_token = top_p_sampling(
                logit,
                self.temperature,
                self.top_p)
        return next_token

    def stop_cond(self, sent, stop_by_quot = False):
        return (sent[-1] == self.vocab.eos_id or
                (stop_by_quot and self.vocab[sent[-1]] == '"'))

    def generate(self, sent, stop_by_quot = False):
        for i in range(self.max_tokens):
            x, am = sent_to_inputs(sent)
            logit = self.decode(x, am)
            logit = self.process_logit(i, logit)
            next_token = self.sample_next_token(logit)
            sent.append(next_token)
            if self.stop_cond(sent, stop_by_quot = stop_by_quot):
                break
        return sent

    def postprocess(self, sent):
        sent = [word for word in sent if word != self.vocab.eos_id]
        return sent

    def __call__(self, sent = None, stop_by_quot = False):
        if sent is None:
            sent = []
        sent = [self.vocab.eos_id] + sent
        sent = self.generate(sent, stop_by_quot = stop_by_quot)
        sent = self.postprocess(sent)
        return sent

