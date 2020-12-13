from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1, start_lr=0.01):
        self.total_iters = total_iters
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        ret = [self.start_lr + (base_lr - self.start_lr) * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
        return ret