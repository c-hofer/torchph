import torch


def histogram_intersection_loss(input: torch.Tensor,
                                target: torch.Tensor,
                                size_average: bool=True,
                                reduce: bool=True,
                                symetric_version: bool=True)->torch.Tensor:
    r"""
    This loss function is based on the `Histogram Intersection` score. 
    
    The output is the *negative* Histogram Intersection Score.

    Args:
        input (Tensor): :math:`(N, B)` where `N = batch size` and `B = number of classes`
        target (Tensor): :math:`(N, B)` where `N = batch size` and `B = number of classes`
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                :attr:`size_average` is set to ``False``, the losses are instead summed
                for each minibatch. Ignored if :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional):
        symetric_version (bool, optional): By default, the symetric version of histogram intersection
                is used. If false the asymetric version is used. Default: ``True``

    Returns: Tensor.

    """
    assert input.size() == target.size(), \
        "input.size() != target.size(): {} != {}!".format(input.size(), target.size())
    assert input.dim() == target.dim() == 2, \
        "input, target must be 2 dimensional. Got dim {} resp. {}".format(input.dim(), target.dim())

    minima = input.min(target)
    summed_minima = minima.sum(dim=1)

    if symetric_version:
        normalization_factor = (input.sum(dim=1)).max(target.sum(dim=1))
    else:
        normalization_factor = target.sum(dim=1)

    loss = summed_minima / normalization_factor

    if reduce:
        loss = sum(loss)

        if size_average:
            loss = loss / input.size(0)

    return -loss
