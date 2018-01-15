import torch


def collection_cascade(input, stop_predicate: callable, function_to_apply: callable):
    if stop_predicate(input):
        return function_to_apply(input)
    elif isinstance(input, list or tuple):
        return [function_to_apply(x) for x in input]
    elif isinstance(input, dict):
        return {k: function_to_apply(v) for k, v in input.items()}
    else:
        raise ValueError('Unknown type collection type. Expected list, tuple, dict but got {}'
                         .format(type(input)))


def cuda_cascade(input, **kwargs):
    return collection_cascade(input,
                              stop_predicate=lambda x: isinstance(x, torch._TensorBase),
                              function_to_apply=lambda x: x.cuda(**kwargs))