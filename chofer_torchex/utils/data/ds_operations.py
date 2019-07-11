import torch
import numpy as np
from typing import Union
import torch.utils.data


def ds_random_subset(
        dataset: torch.utils.data.dataset.Dataset,
        percentage: float=None,
        absolute_size: int=None,
        replace: bool=False):
    r"""
    Represents a fixed random subset of the given dataset.

    Args:
        dataset (torch.utils.data.dataset.Dataset): Target dataset.
        percentage (float): Percentage of target dataset to use (within [0,1]).
        absolute_size (int): Absolute size of the subset to use.
        replace (bool): Draw with replacement.

    Returns:
        A ``torch.utils.data.dataset.Dataset`` with randomly selected samples.

    .. note::
        ``percentage`` and ``absolute_size`` are mutally exclusive. So only
        one of them can be specified.
    """
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    assert percentage is not None or absolute_size is not None
    assert not (percentage is None and absolute_size is None)
    if percentage is not None:
        assert 0 < percentage and percentage < 1, "percentage assumed to be > 0 and < 1"
    if absolute_size is not None:
        assert absolute_size <= len(dataset)

    n_samples = int(percentage*len(dataset)) if percentage is not None else absolute_size
    indices = np.random.choice(
        list(range(len(dataset))),
        n_samples,
        replace=replace)

    indices = [int(i) for i in indices]

    return torch.utils.data.dataset.Subset(dataset, indices)


def ds_label_filter(
        dataset: torch.utils.data.dataset.Dataset,
        labels: Union[tuple, list]):
    """
    Returns a dataset with samples having selected labels.

    Args:
        dataset (torch.utils.data.dataset.Dataset): Target dataset.
        labels (tuple or list): White list of labels to use.

    Returns:
        A ``torch.utils.data.dataset.Dataset`` only containing samples having
        the selected labels.
    """
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    assert isinstance(labels, (tuple, list)), "labels is expected to be list or tuple."
    assert len(set(labels)) == len(labels), "labels is expected to have unique elements."
    assert hasattr(dataset, 'targets'), "dataset is expected to have 'targets' attribute"
    assert set(labels) <= set(dataset.targets), "labels is expected to contain only valid labels of dataset"

    indices = [i for i in range(len(dataset)) if dataset.targets[i] in labels]

    return torch.utils.data.dataset.Subset(dataset, indices)
