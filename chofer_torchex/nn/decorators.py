from torch.autograd import Variable
from torch.nn import Module


class ForwardMemory:
    def __init__(self, module: Module):
        self._enabled = True
        self.memory = None
        self._handle_for_hook = module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output) -> None:
        if not self._enabled:
            return

        if isinstance(output, Variable):
            output = output.data

        if output.is_cuda:
            output = output.cpu()
        else:
            output = output.clone()

        self.memory = output

    @property
    def enabled(self):
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def detach(self):
        self._handle_for_hook.remove()
