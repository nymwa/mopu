import numpy as np
import torch
from soweli.util import generate_square_subsequent_mask
from tunimi.vocabulary import Vocabulary
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

def sent_to_inputs(sent):
    x = torch.tensor([sent]).T
    am = generate_square_subsequent_mask(x.shape[0])
    if torch.cuda.is_available():
        x = x.cuda()
        am = am.cuda()
    return x, am

def process_logit(vocab, logit,
        no_specials = False,
        no_number = False,
        no_proper = False):
    logit[vocab.pad_id] = float('-inf')
    logit[vocab.unk_id] = float('-inf')
    if no_specials or no_number:
        logit[vocab.number_id] = float('-inf')
    if no_specials or no_proper:
        logit[vocab.proper_id] = float('-inf')
    return logit

def load_vocab_and_model(
        hidden_size,
        nhead,
        num_layers,
        checkpoint_path):
    vocab = Vocabulary()
    model = SoweliToki(len(vocab), hidden_size, nhead, hidden_size * 4, num_layers, 0.0, 0.0)
    model.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu'))
    if torch.cuda.is_available():
        model = model.cuda()
    return vocab, model

