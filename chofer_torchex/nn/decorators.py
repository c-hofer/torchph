from torch.autograd import Variable
from torch.nn import Module
from typing import Callable


class OutputMemory:
    def __init__(self, module: Module,
                 get_memory_init: Callable = None,
                 get_updated_memory: Callable = None):
        self._enabled = True

        if get_memory_init is not None:
            self.get_memory_init = get_memory_init

        if get_updated_memory is not None:
            self.get_updated_memory = get_updated_memory

        self.memory = self.get_memory_init()

        self._handle_for_hook = module.register_forward_hook(self.forward_hook)

    @staticmethod
    def get_memory_init():
        return []

    @staticmethod
    def get_updated_memory(memory, val):
        memory.append(val)
        return memory

    def forward_hook(self, module, input, output) -> None:
        if not self._enabled:
            return

        if isinstance(output, Variable):
            output = output.data

        if output.is_cuda:
            output = output.cpu()
        else:
            output = output.clone()

        self.memory = self.get_updated_memory(self.memory, output)

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def reset_memory(self):
        self.memory = self.get_memory_init()

    def enable_and_reset(self):
        self._enabled = True
        self.memory = self.get_memory_init()

    def detach(self):
        self._handle_for_hook.remove()
