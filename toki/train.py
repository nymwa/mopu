from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tunimi.vocabulary import Vocabulary
from .dataset import TokiDataset
from pali.sampler import Sampler
from soweli.toki import SoweliToki

from .task import TokiTask
from .trainer import TokiTrainer

def make_dataset(vocab, dataset_path, valid_size):
    with open(dataset_path) as f:
        sents = [x.strip().split() for x in f]
    sents = [[vocab.indices[token] for token in sent] for sent in sents]
    train_data = sents[:-valid_size]
    valid_data = sents[-valid_size:]
    train_dataset = TokiDataset(train_data, vocab)
    valid_dataset = TokiDataset(valid_data, vocab)
    return train_dataset, valid_dataset

def prepare_loaders(vocab, dataset_path, valid_size, max_tokens):
    train_dataset, valid_dataset = make_dataset(vocab, dataset_path, valid_size)
    train_sampler = Sampler(train_dataset, max_tokens)
    valid_sampler = Sampler(valid_dataset, max_tokens)
    train_loader = DataLoader(train_dataset, batch_sampler = train_sampler, collate_fn = train_dataset.collate)
    valid_loader = DataLoader(valid_dataset, batch_sampler = valid_sampler, collate_fn = valid_dataset.collate)
    return train_loader, valid_loader

def train(dataset_path, checkpoint_path,
        hidden_size, nhead, num_layers,
        attention_dropout, dropout,
        max_tokens, valid_size, epochs):

    vocab = Vocabulary()
    model = SoweliToki(len(vocab), hidden_size, nhead, hidden_size * 4,
            num_layers, attention_dropout, dropout)
    model = model.cuda()
    print('#params (to train): {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('#params (total): {}'.format(sum(p.numel() for p in model.parameters())))
    task = TokiTask(vocab, model,
            lr = 0.002,
            weight_decay = 0.002,
            warmup_steps = 4000,
            clip_norm = 0.3)
    train_loader, valid_loader = prepare_loaders(vocab, dataset_path, valid_size, max_tokens)
    trainer = TokiTrainer(epochs, task, train_loader, valid_loader)
    trainer.train()
    trainer.save(checkpoint_path)

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='preprocessed.txt')
    parser.add_argument('--checkpoint', default='checkpoint.pt')
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--attention-dropout', type=float, default=0.2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max-tokens', type=int, default=4000)
    parser.add_argument('--valid-size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    train(args.dataset,
            args.checkpoint,
            args.hidden_size,
            args.nhead,
            args.num_layers,
            args.attention_dropout,
            args.dropout,
            args.max_tokens, 
            args.valid_size,
            args.epochs)

