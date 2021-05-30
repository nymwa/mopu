import torch
import torch.nn as nn
import torch.optim as optim
from apex import amp
from pali.scheduler import WarmupScheduler

class TokiTask:
    def __init__(self, vocab, model, lr, weight_decay, warmup_steps, clip_norm):
        self.vocab = vocab
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level = 'O1')
        self.scheduler = WarmupScheduler(self.optimizer, warmup_steps)
        self.criterion = nn.CrossEntropyLoss(ignore_index = vocab.pad_id)
        self.clip_norm = clip_norm

    def calculate_loss(self, batch):
        pred = self.model(batch)
        pred = pred.view(-1, pred.size(-1))
        loss = self.criterion(pred, batch.outputs.view(-1))
        return loss

    def train_step(self, accum, batch):
        batch.cuda()
        self.optimizer.zero_grad()
        loss = self.calculate_loss(batch)
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        grad = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        accum.update(batch, loss, lr, grad)

    def valid_step(self, accum, batch):
        batch.cuda()
        with torch.no_grad():
            loss = self.calculate_loss(batch)
        accum.update(batch, loss)

