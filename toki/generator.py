import numpy as np
import torch
from tunimi.vocabulary import Vocabulary
from tunimi.proper import ProperGenerator
from soweli.util import generate_square_subsequent_mask
from soweli.toki import SoweliToki

def top_p_sampling(logit, temperature, top_p):
    logit = logit / temperature
    probs = torch.softmax(logit, dim = -1)
    values, indices = torch.sort(probs)
    cumlated = torch.cumsum(values, -1)
    is_removed = cumlated < (1 - top_p)
    logit[indices[is_removed]] = -float('Inf')
    probs = torch.softmax(logit, dim = -1)
    probs = probs.cpu().numpy()
    next_token = np.random.choice(range(len(probs)), p=probs)
    return next_token

def sample_sentence(vocab, model, temperature = 1.0, top_p = 0.8, max_tokens = 64, stop_ratio = 0.5, no_specials = False):
    sent = [vocab.eos_id]
    for i in range(max_tokens):
        x = torch.tensor([sent]).T
        am = generate_square_subsequent_mask(x.shape[0])
        if torch.cuda.is_available():
            x = x.cuda()
            am = am.cuda()
        with torch.no_grad():
            y = model.decode(x, attention_mask = am)
        logit = y[-1, 0, :]
        logit[vocab.pad_id] = float('-inf')
        logit[vocab.unk_id] = float('-inf')
        if no_specials:
            logit[vocab.number_id] = float('-inf')
            logit[vocab.proper_id] = float('-inf')
        if i > max_tokens * stop_ratio:
            logit[vocab.eos_id] = logit[vocab.eos_id] * (i - max_tokens * stop_ratio + 1)
        next_token = top_p_sampling(y[-1, 0, :], temperature, top_p)
        sent.append(next_token)
        if sent[-1] == vocab.eos_id:
            break
    return sent[1:-1]

class TokiGenerator:
    def __init__(self, checkpoint, hidden_size, nhead, num_layers):
        self.vocab = Vocabulary()
        self.model = SoweliToki(len(self.vocab),
                hidden_size,
                nhead,
                hidden_size * 4,
                num_layers,
                0.0, 0.0)
        self.model.load_state_dict(torch.load(checkpoint, map_location = 'cpu'))
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def __call__(self,
            as_str = False,
            no_specials = False,
            temperature = 1.0,
            top_p = 0.8,
            max_tokens = 64,
            stop_ratio = 0.5):
        x = sample_sentence(
                self.vocab,
                self.model,
                temperature = temperature,
                top_p = top_p,
                max_tokens = max_tokens,
                stop_ratio = stop_ratio,
                no_specials = no_specials)
        if as_str:
            x = [self.vocab[n] for n in x]
        return x

class ProperTransformer:
    def __init__(self):
        self.generator = ProperGenerator()

    def __call__(self, sent):
        indices = [i for i, x in enumerate(sent) if x == '<proper>']
        proper_list = [self.generator() for _ in range(len(indices))]
        for index in indices:
            n = np.random.binomial(len(indices) - 1, 0.5)
            sent[index] = proper_list[n]
        return sent

class SoweliConverter:
    def __call__(self, source):
        target = []
        for token in source:
            if token == 'jan':
                target.append('soweli')
            elif token == 'soweli':
                target.append('jan')
            else:
                target.append(token)
        return target

