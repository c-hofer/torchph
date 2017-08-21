from .plugin import Plugin


class LearningRateScheduler(Plugin):
    def __init__(self, learning_rate_shedule: callable):
        self.learning_rate_schedule = learning_rate_shedule
        self._trainer = None
        self._learning_rate = None

    def register(self, trainer):
        self._trainer = trainer

        self._trainer.events.pre_epoch.append(self.pre_epoch_handler)

    def set_learning_rate(self, lr):
        optimizer = self._trainer.optimizer

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def pre_epoch_handler(self, *args, **kwargs):
        new_learning_rate = self.learning_rate_schedule(self._trainer)

        if new_learning_rate is not None:
            self.set_learning_rate(new_learning_rate)
            self._learning_rate = new_learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate