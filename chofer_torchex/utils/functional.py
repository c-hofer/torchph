# TODO: Possibly delete

import torch


def collection_cascade(input, stop_predicate: callable, function_to_apply: callable):
    if stop_predicate(input):
        return function_to_apply(input)
    elif isinstance(input, list or tuple):
        return [collection_cascade(
            x,
            stop_predicate=stop_predicate,
            function_to_apply=function_to_apply) for x in input]
    elif isinstance(input, dict):
        return {k: collection_cascade(
            v,
            stop_predicate=stop_predicate,
            function_to_apply=function_to_apply) for k, v in input.items()}
    else:
        raise ValueError('Unknown type collection type. Expected list, \
                          tuple, dict but got {}'.format(type(input)))


def cuda_cascade(input, **kwargs):
    return collection_cascade(
        input,
        stop_predicate=lambda x: isinstance(x, torch.Tensor),
        function_to_apply=lambda x: x.cuda(**kwargs))
