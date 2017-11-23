import torch
from torch.autograd import Variable
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable
from torch import Tensor



class Event:
    def __init__(self):
        self._callbacks = []

    def __call__(self, default_kwargs: {}, **kwargs):
        kwargs.update(default_kwargs)
        for callback in self._callbacks:
            callback(**kwargs)

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
    def __init__(self, model: Module, loss: Callable, optimizer: Optimizer, train_data: DataLoader, n_epochs,
                 cuda=False,
                 cuda_device_id=None,
                 variable_created_by_model=False):
        self.n_epochs = n_epochs
        self.model = model
        self.criterion = loss
        self.optimizer = optimizer
        self.train_data = train_data
        self.epoch_count = 1
        self.cuda = cuda
        self.cuda_device_id = cuda_device_id
        self.variable_created_by_model = variable_created_by_model

        self.return_value = {}

        self.events = TrainerEvents()

    def _get_default_event_kwargs(self):
        return {
                'trainer': self,
                'model': self.model,
                'epoch_count': self.epoch_count,
                'cuda': self.cuda
                }

    @property
    def iteration_count(self):
        return self.batch_count * self.epoch_count

    def register_plugin(self, plugin):
        plugin.register(self)

    def run(self):
        self.events.pre_run(self._get_default_event_kwargs())

        if self.cuda:
            self.model.cuda(self.cuda_device_id)

        for i in range(1, self.n_epochs + 1):
            self.epoch_count = i

            self.events.pre_epoch(self._get_default_event_kwargs(),
                                  optimizer=self.optimizer,
                                  train_data=self.train_data,
                                  max_epochs=self.n_epochs,
                                  current_epoch_number=i)
            self._train_epoch()
            self.events.post_epoch(self._get_default_event_kwargs(), trainer=self)
            self.events.post_epoch_gui(self._get_default_event_kwargs())

        self.events.post_run(self._get_default_event_kwargs())
        return self.return_value

    def _train_epoch(self):
        self.model.train()

        for i, (batch_input, batch_target) in enumerate(self.train_data, start=1):

            self.events.pre_batch(self._get_default_event_kwargs(),
                                  batch_input=batch_input,
                                  batch_target=batch_target)

            batch_input, batch_target = self.data_typing(batch_input, batch_target)

            target_var = Variable(batch_target)

            if not self.variable_created_by_model:
                batch_input = Variable(batch_input)

            def closure():
                batch_output = self.model(batch_input)
                loss = self.criterion(batch_output, target_var)
                loss.backward()

                assert len(loss.data) == 1
                self.events.post_batch_backward(self._get_default_event_kwargs(),
                                                batch_output=batch_output,
                                                loss=float(loss.data[0]))

                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)

            self.events.post_batch(self._get_default_event_kwargs(),
                                   batch_input=batch_input,
                                   batch_target=batch_target,
                                   current_batch_number=i)

    @staticmethod
    def before_data_typing_hook(batch_input, batch_targets):
        return batch_input, batch_targets

    @staticmethod
    def after_data_typing_hook(batch_input, batch_targets):
        return batch_input, batch_targets

    def data_typing(self, batch_input, batch_targets):
        batch_input, batch_targets = self.before_data_typing_hook(batch_input, batch_targets)

        tensor_cast = Tensor.cuda if self.cuda else Tensor.cpu

        def cast(x):
            if isinstance(x, torch.tensor._TensorBase):
                return tensor_cast(x)
            elif isinstance(x, list):
                return [cast(v) for v in x]
            elif isinstance(x, dict):
                return {k: cast(v) for k, v in x.items()}
            elif isinstance(x, tuple):
                return tuple(cast(v) for v in x)
            else:
                return x

        batch_input = cast(batch_input)
        batch_targets = cast(batch_targets)

        batch_input, batch_targets = self.after_data_typing_hook(batch_input, batch_targets)

        return batch_input, batch_targets
