class Accumulator:
    def __init__(self, phase, epoch, num_batches):
        self.phase = phase
        self.epoch = epoch
        self.num_batches = num_batches
        self.num_list = []
        self.loss_list = []
        self.wpb_list = []
        self.spb_list = []
        self.lr_list = []
        self.grad_list = []

    def update(self, batch, loss, lr = None, grad = None):
        self.num_list.append(len(batch))
        self.loss_list.append(loss.item())
        self.wpb_list.append(sum(batch.get_lengths()))
        self.spb_list.append(len(batch.get_lengths()))
        if lr is not None:
            self.lr_list.append(lr)
        if grad is not None:
            self.grad_list.append(grad)

    def step_log(self):
        line = '| inner | epoch {}, {}/{} | loss {:.4f} | w/b {} | s/b {}'.format(
                self.epoch,
                len(self.num_list),
                self.num_batches,
                self.loss_list[-1],
                self.wpb_list[-1],
                self.spb_list[-1])
        if self.lr_list:
            line += ' | lr {:.4e}'.format(self.lr_list[-1])
        if self.grad_list:
            line += ' | grad {:.4f}'.format(self.grad_list[-1])
        return line

    def avg(self, lst):
        num_examples = sum(self.num_list)
        return sum([n * x for n, x in zip(self.num_list, lst)]) / num_examples

    def epoch_log(self, num_steps = None):
        line = '| {} | epoch {} | loss {:.4f} | w/b {:.1f} | s/b {:.1f}'.format(
                self.phase,
                self.epoch,
                self.avg(self.loss_list),
                self.avg(self.lr_list),
                self.avg(self.wpb_list),
                self.avg(self.spb_list))
        if num_steps is not None:
            line += ' | steps {}'.format(num_steps)
        if self.lr_list:
            line += ' | lr {:.4e}'.format(self.avg(self.lr_list))
        if self.grad_list:
            line += ' | grad {:.4f}'.format(self.avg(self.grad_list))
        return line

