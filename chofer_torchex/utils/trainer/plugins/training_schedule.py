from .plugin import Plugin
from collections import OrderedDict


class LearningRateScheduler(Plugin):
    def __init__(self, learning_rate_shedule: callable):
        super(LearningRateScheduler, self).__init__()
        self.learning_rate_schedule = learning_rate_shedule
        self.learning_rate_by_epoch = OrderedDict()
        self.optimizer = None

    def register(self, trainer):
        self.optimizer = trainer.optimizer
        trainer.events.pre_epoch.append(self.pre_epoch_handler)

    def set_optimizer_to_new_lr(self, lr, optimizer):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def pre_epoch_handler(self, **kwargs):
        optimizer = kwargs['optimizer']

        new_learning_rate = self.learning_rate_schedule(**kwargs)

        if new_learning_rate is not None:
            self.set_optimizer_to_new_lr(new_learning_rate, optimizer)
            self._learning_rate['n_epochs'] = new_learning_rate
