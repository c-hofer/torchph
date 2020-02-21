import torch
import torch.nn as nn


class LinearView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Apply(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class IndependentBranchesLinear(nn.Linear):
    def __init__(self, in_features, out_features_branch, n_branches, bias=True):
        assert in_features % n_branches == 0
        in_features_branch = int(in_features/n_branches)
        super().__init__(in_features, out_features_branch*n_branches, bias)
        
        mask = torch.zeros_like(self.weight)
        for i in range(n_branches):
            mask[i*out_features_branch:(i+1)*out_features_branch,
                 i*in_features_branch:(i+1)*in_features_branch] = 1

        self.register_buffer('mask', mask)

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight * self.mask, self.bias)


class View(nn.Module):
    def __init__(self, view_args):
        super().__init__()
        self.view_args = view_args

    def forward(self, input):
        return input.view(*self.view_args)
