import torch
from pali.accumulator import Accumulator

from pali.log import init_logging
from logging import getLogger
init_logging()
logger = getLogger(__name__)

class TokiTrainer:
    def __init__(self, epochs, task, train_loader, valid_loader):
        self.epochs = epochs
        self.task = task
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_steps = 0

    def train_epoch(self, epoch):
        self.task.model.train()
        accum = Accumulator('train', epoch, len(self.train_loader))
        for step, batch in enumerate(self.train_loader):
            self.task.train_step(accum, batch)
            self.num_steps += 1
            logger.info(accum.step_log())
        logger.info(accum.epoch_log(self.num_steps))

    def valid_epoch(self, epoch):
        self.task.model.eval()
        accum = Accumulator('valid', epoch, len(self.valid_loader))
        for step, batch in enumerate(self.valid_loader):
            self.task.valid_step(accum, batch)
        logger.info(accum.epoch_log(self.num_steps))

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)
            self.valid_epoch(epoch)

    def save(self, checkpoint_path):
        torch.save(self.task.model.state_dict(), checkpoint_path)

