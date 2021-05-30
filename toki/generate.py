import sys
import numpy as np
import torch
from tunimi.vocabulary import Vocabulary
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

def sample_sentence(vocab, model, temperature = 1.0, top_p = 0.8, max_tokens = 64, stop_ratio = 0.5):
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
        if i > max_tokens * stop_ratio:
            logit[vocab.eos_id] = logit[vocab.eos_id] * (i - max_tokens * stop_ratio + 1)
        next_token = top_p_sampling(y[-1, 0, :], temperature, top_p)
        sent.append(next_token)
        if sent[-1] == vocab.eos_id:
            break
    return sent[1:-1]

def main():
    hidden_size = 512
    num_layers = 6
    nhead = 8

    vocab = Vocabulary()
    model = SoweliToki(len(vocab), hidden_size, nhead, hidden_size * 4,
            num_layers, 0.0, 0.0)
    model.load_state_dict(torch.load('checkpoint.pt', map_location = 'cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    while True:
        x = sample_sentence(vocab, model)
        x = ' '.join([vocab[n] for n in x])
        print(x)

