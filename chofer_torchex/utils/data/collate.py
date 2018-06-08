from collections import defaultdict


def dict_sample_target_iter_concat(sample_target_iter: iter):
    """
    Gets an sample target iterator of dict samples. Returns
    a concatenation of the samples based on each key and the
    target list.

    Example:
    ```
    sample_target_iter = [({'a': 'a1', 'b': 'b1'}, 0), ({'a': 'a2', 'b': 'b2'}, 1)]
    x = dict_sample_iter_concat([({'a': 'a1', 'b': 'b1'}, 0), ({'a': 'a2', 'b': 'b2'}, 1)])
    print(x)
    ({'a': ['a1', 'a2'], 'b': ['b1', 'b2']}, [0, 1])
    ```

    :param sample_target_iter:
    :return:
    """

    samples = defaultdict(list)
    targets = []

    for sample_dict, y in sample_target_iter:
        for k, v in sample_dict.items():
            samples[k].append(v)

        targets.append(y)

    samples = dict(samples)

    length = len(samples[next(iter(samples))])
    assert all(len(samples[k]) == length for k in samples)

    return samples, targets



