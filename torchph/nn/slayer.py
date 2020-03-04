r"""

Implementation of **differentiable vectorization layers** for persistent
homology barcodes.

For a basic tutorial click `here <tutorials/SLayer.html>`_.
"""
import torch
import numpy as np
from torch.tensor import Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from typing import List, Tuple

import warnings


# region helper functions


def prepare_batch(
        batch: List[Tensor],
        point_dim: int=None
        )->Tuple[Tensor, Tensor, int, int]:
    """
    This method 'vectorizes' the multiset in order to take advances of GPU
    processing. The policy is to embed all multisets in batch to the highest
    dimensionality occurring in batch, i.e., ``max(t.size()[0]`` for ``t`` in batch).

    Args:
        batch:
            The input batch to process as a list of tensors.

        point_dim:
            The dimension of the points the inputs consist of.

    Returns:
        A four-tuple consisting of (1) the constructed ``batch``, i.e., a
        tensor with size
        ``batch_size`` x ``n_max_points`` x ``point_dim``; (2) a tensor
        ``not_dummy`` of size ``batch_size`` x ``n_max_points``, where
        ``1`` at position (i,j) indicates if the point is a dummy point,
        whereas ``0`` indicates a dummy point used for padding; (3)
        the max. number of points and (4) the batch size.

    Example::

        >>> from torchph.nn.slayer import prepare_batch
        >>> import torch
        >>> x = [torch.rand(10,2), torch.rand(20,2)]
        >>> batch, not_dummy, max_pts, batch_size = prepare_batch(x)
    """
    if point_dim is None:
        point_dim = batch[0].size(1)
    assert (all(x.size(1) == point_dim for x in batch if len(x) != 0))

    batch_size = len(batch)
    batch_max_points = max([t.size(0) for t in batch])
    input_device = batch[0].device

    if batch_max_points == 0:
        # if we are here, batch consists only of empty diagrams.
        batch_max_points = 1

    # This will later be used to set the dummy points to zero in the output.
    not_dummy_points = torch.zeros(
        batch_size,
        batch_max_points,
        device=input_device)

    prepared_batch = []

    for i, multi_set in enumerate(batch):
        n_points = multi_set.size(0)

        prepared_dgm = torch.zeros(
            batch_max_points,
            point_dim,
            device=input_device)

        if n_points > 0:
            index_selection = torch.tensor(range(n_points),
                                           device=input_device)

            prepared_dgm.index_add_(0, index_selection, multi_set)

            not_dummy_points[i, :n_points] = 1

        prepared_batch.append(prepared_dgm)

    prepared_batch = torch.stack(prepared_batch)

    return prepared_batch, not_dummy_points, batch_max_points, batch_size


def is_prepared_batch(input):
    if not (isinstance(input, tuple) and len(input) == 4):
        return False
    else:
        batch, not_dummy_points, max_points, batch_size = input
        return isinstance(batch, Tensor) and isinstance(not_dummy_points, Tensor) and max_points > 0 and batch_size > 0


def is_list_of_tensors(input):
    try:
        return all([isinstance(x, Tensor) for x in input])

    except TypeError:
        return False


def prepare_batch_if_necessary(input, point_dimension=None):
    batch, not_dummy_points, max_points, batch_size = None, None, None, None

    if is_prepared_batch(input):
        batch, not_dummy_points, max_points, batch_size = input
    elif is_list_of_tensors(input):
        if point_dimension is None:
            point_dimension = input[0].size(1)

        batch, not_dummy_points, max_points, batch_size = prepare_batch(
            input,
            point_dimension)

    else:
        raise ValueError(
            'SLayer does not recognize input format! Expecting [Tensor] or \
             prepared batch. Not {}'.format(input))

    return batch, not_dummy_points, max_points, batch_size


def parameter_init_from_arg(arg, size, default, scalar_is_valid=False):
    if isinstance(arg, (int, float)):
        if not scalar_is_valid:
            raise ValueError('Scalar initialization values are not valid. \
                              Got {} expected Tensor of size {}.'
                             .format(arg, size))
        return torch.Tensor(*size).fill_(arg)
    elif isinstance(arg, torch.Tensor):
        assert(arg.size() == size)
        return arg
    elif arg is None:
        if default in [torch.rand, torch.randn, torch.ones, torch.ones_like]:
            return default(*size)
        else:
            return default(size)
    else:
        raise ValueError('Cannot handle parameter initialization. \
                          Got "{}" '.format(arg))

# endregion


class SLayerExponential(Module):
    """
    Proposed input layer for multisets [1].
    """
    def __init__(self, n_elements: int,
                 point_dimension: int=2,
                 centers_init: Tensor=None,
                 sharpness_init: Tensor=None):
        """
        Args:
            n_elements:
                Number of structure elements used.

            point_dimension: D
                Dimensionality of the points of which the
                input multi set consists of.

            centers_init:
                The initialization for the centers of the structure elements.

            sharpness_init:
                Initialization for the sharpness of the structure elements.
        """
        super().__init__()

        self.n_elements = n_elements
        self.point_dimension = point_dimension

        expected_init_size = (self.n_elements, self.point_dimension)

        centers_init = parameter_init_from_arg(
            centers_init,
            expected_init_size,
            torch.rand, scalar_is_valid=False)
        sharpness_init = parameter_init_from_arg(
            sharpness_init,
            expected_init_size,
            lambda size: torch.ones(*size)*3)

        self.centers = Parameter(centers_init)
        self.sharpness = Parameter(sharpness_init)

    def forward(self, input)->Tensor:
        batch, not_dummy_points, max_points, batch_size = prepare_batch_if_necessary(
            input,
            point_dimension=self.point_dimension)

        batch = torch.cat([batch] * self.n_elements, 1)

        not_dummy_points = torch.cat([not_dummy_points] * self.n_elements, 1)

        centers = torch.cat([self.centers] * max_points, 1)
        centers = centers.view(-1, self.point_dimension)
        centers = torch.stack([centers] * batch_size, 0)

        sharpness = torch.pow(self.sharpness, 2)
        sharpness = torch.cat([sharpness] * max_points, 1)
        sharpness = sharpness.view(-1, self.point_dimension)
        sharpness = torch.stack([sharpness] * batch_size, 0)

        x = centers - batch
        x = x.pow(2)
        x = torch.mul(x, sharpness)
        x = torch.sum(x, 2)
        x = torch.exp(-x)
        x = torch.mul(x, not_dummy_points)
        x = x.view(batch_size, self.n_elements, -1)
        x = torch.sum(x, 2)
        x = x.squeeze()

        return x

    def __repr__(self):
        return 'SLayerExponential (... -> {} )'.format(self.n_elements)


class SLayerRational(Module):
    """
    """
    def __init__(self, n_elements: int,
                 point_dimension: int=2,
                 centers_init: Tensor=None,
                 sharpness_init: Tensor=None,
                 exponent_init: Tensor=None,
                 pointwise_activation_threshold=None,
                 share_sharpness=False,
                 share_exponent=False,
                 freeze_exponent=True):
        """
        Args:
            n_elements:
                Number of structure elements used.

            point_dimension:
                Dimensionality of the points of which the input multi set
                consists of.

            centers_init:
                The initialization for the centers of the structure elements.

            sharpness_init:
                Initialization for the sharpness of the structure elements.
        """
        super().__init__()

        self.n_elements = int(n_elements)
        self.point_dimension = int(point_dimension)
        self.pointwise_activation_threshold = float(pointwise_activation_threshold) \
            if pointwise_activation_threshold is not None else None
        self.share_sharpness = bool(share_sharpness)
        self.share_exponent = bool(share_exponent)
        self.freeze_exponent = freeze_exponent

        if self.pointwise_activation_threshold is not None:
            self.pointwise_activation_threshold = float(self.pointwise_activation_threshold)

        centers_init = parameter_init_from_arg(
            arg=centers_init,
            size=(self.n_elements, self.point_dimension),
            default=torch.rand)

        sharpness_init = parameter_init_from_arg(
            arg=sharpness_init,
            size=(1,) if self.share_sharpness else (self.n_elements, self.point_dimension),
            default=torch.ones,
            scalar_is_valid=True)

        exponent_init = parameter_init_from_arg(
            arg=exponent_init,
            size=(1,) if self.share_exponent else (self.n_elements,),
            default=torch.ones,
            scalar_is_valid=True)

        self.centers = Parameter(centers_init)
        self.sharpness = Parameter(sharpness_init)

        if self.freeze_exponent:
            self.register_buffer('exponent', exponent_init)
        else:
            self.exponent = Parameter(exponent_init)

    def forward(self, input)->Tensor:
        batch, not_dummy_points, max_points, batch_size = prepare_batch_if_necessary(
            input,
            point_dimension=self.point_dimension)

        batch = batch.unsqueeze(1).expand(
            batch_size,
            self.n_elements,
            max_points,
            self.point_dimension)

        not_dummy_points = not_dummy_points.unsqueeze(1).expand(-1, self.n_elements, -1)

        centers = self.centers.unsqueeze(1).expand(
            self.n_elements,
            max_points,
            self.point_dimension)
        centers = centers.unsqueeze(0).expand(batch_size, *centers.size())

        sharpness = self.sharpness
        if self.share_sharpness:
            sharpness = sharpness.expand(self.n_elements, self.point_dimension)
        sharpness = sharpness.unsqueeze(1).expand(-1, max_points, -1)
        sharpness = sharpness.unsqueeze(0).expand(batch_size, *sharpness.size())

        exponent = self.exponent
        if self.share_exponent:
            exponent = exponent.expand(self.n_elements)

        exponent = exponent.unsqueeze(1).expand(-1, max_points)
        exponent = exponent.unsqueeze(0).expand(batch_size, *exponent.size())

        x = centers - batch
        x = x.abs()
        x = torch.mul(x, sharpness.abs())
        x = torch.sum(x, 3)
        x = 1/(1+x).pow(exponent.abs())

        if self.pointwise_activation_threshold is not None:
            x[(x < self.pointwise_activation_threshold).data] = 0

        x = torch.mul(x, not_dummy_points)
        x = torch.sum(x, 2)

        return x

    def __repr__(self):
        return 'SLayerRational (... -> {} )'.format(self.n_elements)


class SLayerRationalHat(Module):
    """
    """
    def __init__(self, n_elements: int,
                 point_dimension: int=2,
                 centers_init: Tensor=None,
                 radius_init: float=1,
                 exponent: int=1
                 ):
        """
        Args:
            n_elements:
                Number of structure elements used.

            point_dimension:
                Dimensionality of the points of which the input multi
                set consists of.

            centers_init:
                The initialization for the centers of the structure elements.

            radius_init:
                Initialization for radius of zero level-set of the hat.

            exponent:
                Exponent of the rationals forming the hat.

        """
        super().__init__()

        self.n_elements = int(n_elements)
        self.point_dimension = int(point_dimension)
        self.exponent = int(exponent)

        centers_init = parameter_init_from_arg(arg=centers_init,
                                               size=(self.n_elements, self.point_dimension),
                                               default=torch.rand)

        radius_init = parameter_init_from_arg(arg=radius_init,
                                              size=(self.n_elements,),
                                              default=torch.ones,
                                              scalar_is_valid=True)

        self.centers = Parameter(centers_init)
        self.radius = Parameter(radius_init)
        self.norm_p = 1

    def forward(self, input)->Tensor:
        batch, not_dummy_points, max_points, batch_size = prepare_batch_if_necessary(
            input,
            point_dimension=self.point_dimension)

        batch = batch.unsqueeze(1).expand(
            batch_size,
            self.n_elements,
            max_points,
            self.point_dimension)

        not_dummy_points = not_dummy_points.unsqueeze(1).expand(-1, self.n_elements, -1)

        centers = self.centers.unsqueeze(1).expand(
            self.n_elements,
            max_points,
            self.point_dimension)
        centers = centers.unsqueeze(0).expand(batch_size, *centers.size())

        radius = self.radius
        radius = radius.unsqueeze(1).expand(-1, max_points)
        radius = radius.unsqueeze(0).expand(batch_size, *radius.size())
        radius = radius.abs()

        norm_to_center = centers - batch
        norm_to_center = torch.norm(norm_to_center, p=self.norm_p, dim=3)

        positive_part = 1/(1+norm_to_center).pow_(self.exponent)

        negative_part = 1/(1 + (radius - norm_to_center).abs_()).pow_(self.exponent)

        x = positive_part - negative_part

        x = torch.mul(x, not_dummy_points)
        x = torch.sum(x, 2)

        return x

    def __repr__(self):
        return 'SLayerRationalHat (... -> {} )'.format(self.n_elements)


class LinearRationalStretchedBirthLifeTimeCoordinateTransform:
    def __init__(self, nu):
        self._nu = nu
        self._nu_squared = nu**2
        self._2_nu = 2*nu

    def __call__(self, dgm):
        if len(dgm) == 0:
            return dgm

        x, y = dgm[:, 0], dgm[:, 1]
        y = y - x

        i = (y <= self._nu)
        y[i] = - self._nu_squared/y[i] + self._2_nu

        return torch.stack([x, y], dim=1)


class LogStretchedBirthLifeTimeCoordinateTransform:
    def __init__(self, nu):
        self.nu = nu

    def __call__(self, dgm):
        if len(dgm) == 0:
            return dgm

        x, y = dgm[:, 0], dgm[:, 1]
        y = y - x

        i = (y <= self.nu)
        y[i] = torch.log(y[i] / self.nu)*self.nu + self.nu

        return torch.stack([x, y], dim=1)


class UpperDiagonalThresholdedLogTransform:
    def __init__(self, nu):
        self.b_1 = (torch.Tensor([1, 1]) / np.sqrt(2))
        self.b_2 = (torch.Tensor([-1, 1]) / np.sqrt(2))
        self.nu = nu

    def __call__(self, dgm):
        if len(dgm) == 0:
            return dgm

        self.b_1 = self.b_1.to(dgm.device)
        self.b_2 = self.b_2.to(dgm.device)

        x = torch.mul(dgm, self.b_1.repeat(dgm.size(0), 1))
        x = torch.sum(x, 1).squeeze()
        y = torch.mul(dgm, self.b_2.repeat(dgm.size(0), 1))
        y = torch.sum(y, 1).squeeze()
        i = (y <= self.nu)
        y[i] = torch.log(y[i] / self.nu)*self.nu + self.nu
        ret = torch.stack([x, y], 1)
        return ret
