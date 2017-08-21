from torch.autograd import Variable


class Event:
    def __init__(self):
        self._callbacks = []

    def __call__(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)

    def append(self, callback):
        if not callable(callback):
            raise ValueError('Expected callable.')

        self._callbacks.append(callback)


class TrainerEvents:
    def __init__(self):
        self.pre_run = Event()

        self.pre_epoch = Event()
        self.post_epoch = Event()
        self.post_batch_backward = Event()
        self.pre_batch = Event()
        self.post_batch = Event()

        self.post_epoch_gui = Event()

        self.post_run = Event()


class Trainer(object):
    def __init__(self, n_epochs=None, model=None, criterion=None, optimizer=None, dataset=None):
        self.n_epochs = n_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.epoch_count = 1
        self.batch_count = 1

        self.return_value = {}

        self.events = TrainerEvents()

    @property
    def iteration_count(self):
        return self.batch_count * self.batch_count

    def register_plugin(self, plugin):
        plugin.register(self)

    def run(self):
        self.events.pre_run()

        for i in range(1, self.n_epochs + 1):
            self.epoch_count = i

            self.events.pre_epoch()
            self._train()
            self.events.post_epoch()
            self.events.post_epoch_gui()

        self.events.post_run()
        return self.return_value

    def _train(self):
        self.model.train()

        for i, data in enumerate(self.dataset, start=1):
            self.batch_count = i

            batch_input, batch_target = data

            self.events.pre_batch(batch_input=batch_input, batch_target=batch_target)

            target_var = Variable(batch_target)

            def closure():
                batch_output = self.model(batch_input)
                loss = self.criterion(batch_output, target_var)
                loss.backward()

                self.events.post_batch_backward(batch_output=batch_output, loss=loss)

                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)

            self.events.post_batch(batch_input=batch_input, batch_target=batch_target)
