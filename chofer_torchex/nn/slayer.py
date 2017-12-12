import torch
from torch import Tensor, LongTensor
from torch.tensor import _TensorBase
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def safe_tensor_size(tensor, dim):
    try:
        return tensor.size(dim)

    except Exception:
        return 0


class SLayer(Module):
    """
    Implementation of the in

    {
      Hofer17c,
      author    = {C.~Hofer and R.~Kwitt and M.~Niethammer and A.~Uhl},
      title     = {Deep Learning with Topological Signatures},
      booktitle = {NIPS},
      year      = 2017,
      note      = {accepted}
    }

    proposed input layer for multisets.
    """
    def __init__(self, n_elements: int,
                 point_dimension: int=2,
                 centers_init: Tensor=None,
                 sharpness_init: Tensor=None):
        """
        :param n_elements: number of structure elements used
        :param point_dimension: dimensionality of the points of which the input multi set consists of
        :param centers_init: the initialization for the centers of the structure elements
        :param sharpness_init: initialization for the sharpness of the structure elements
        """
        super(SLayer, self).__init__()

        self.n_elements = n_elements
        self.point_dimension = point_dimension

        if centers_init is None:
            centers_init = torch.rand(self.n_elements, self.point_dimension)

        if sharpness_init is None:
            sharpness_init = torch.ones(self.n_elements, self.point_dimension)*3

        self.centers = Parameter(centers_init)
        self.sharpness = Parameter(sharpness_init)

    @staticmethod
    def prepare_batch(batch: [Tensor], point_dim: int)->tuple:
        """
        This method 'vectorizes' the multiset in order to take advances of gpu processing.
        The policy is to embed the all multisets in batch to the highest dimensionality
        occurring in batch, i.e., max(t.size()[0] for t in batch).
        :param batch:
        :param point_dim:
        :return:
        """
        input_is_cuda = batch[0].is_cuda
        assert all(t.is_cuda == input_is_cuda for t in batch)

        # We do the following on cpu since there is a lot of looping
        batch = [x.cpu() for x in batch]

        batch_size = len(batch)
        batch_max_points = max([safe_tensor_size(t, 0) for t in batch])
        input_type = type(batch[0])

        if batch_max_points == 0:
            # if we are here, batch consists only of empty diagrams.
            batch_max_points = 1

        # This will later be used to set the dummy points to zero in the output.
        not_dummy_points = input_type(batch_size, batch_max_points)
        # In the initialization every point is a dummy point.
        not_dummy_points[:, :] = 0

        prepared_batch = []

        for i, multi_set in enumerate(batch):
            n_points = safe_tensor_size(multi_set, 0)

            prepared_dgm = type(multi_set)()
            torch.zeros(batch_max_points, point_dim, out=prepared_dgm)

            if n_points > 0:
                index_selection = LongTensor(range(n_points))

                prepared_dgm.index_add_(0, index_selection, multi_set)

                not_dummy_points[i, :n_points] = 1

            prepared_batch.append(prepared_dgm)

        prepared_batch = torch.stack(prepared_batch)

        if input_is_cuda:
            not_dummy_points = not_dummy_points.cuda()
            prepared_batch = prepared_batch.cuda()

        return prepared_batch, not_dummy_points, batch_max_points, batch_size

    @staticmethod
    def is_prepared_batch(input):
        if not (isinstance(input, tuple) and len(input) == 4):
            return False

        else:
            batch, not_dummy_points, max_points, batch_size = input
            return isinstance(batch, _TensorBase) and isinstance(not_dummy_points, _TensorBase) and max_points > 0 and batch_size > 0

    @staticmethod
    def is_list_of_tensors(input):
        try:
            return all([isinstance(x, _TensorBase) for x in input])

        except TypeError:
            return False

    @staticmethod
    def is_list_of_variables(input):
        try:
            return all(isinstance(x, Variable) for x in input)

        except TypeError:
            return False

    @property
    def is_gpu(self):
        return self.centers.is_cuda

    def forward(self, input)->Variable:
        batch, not_dummy_points, max_points, batch_size = None, None, None, None

        if self.is_prepared_batch(input):
            batch, not_dummy_points, max_points, batch_size = input
        elif self.is_list_of_tensors(input):
            batch, not_dummy_points, max_points, batch_size = SLayer.prepare_batch(input,
                                                                                   self.point_dimension)
        elif self.is_list_of_variables(input):
            input = [x.data for x in input]
            batch, not_dummy_points, max_points, batch_size = SLayer.prepare_batch(input,
                                                                                   self.point_dimension)
        else:
            raise ValueError('SLayer does not recognize input format! Expecting [Tensor] or prepared batch. Not {}'.format(input))

        batch = Variable(batch, requires_grad=False)
        batch = torch.cat([batch] * self.n_elements, 1)

        not_dummy_points = Variable(not_dummy_points, requires_grad=False)
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
        print(x.size())
        x = torch.mul(x, not_dummy_points)
        x = x.view(batch_size, self.n_elements, -1)
        x = torch.sum(x, 2)
        x = x.squeeze()

        return x

    def __str__(self):
        return 'SLayer (... -> {} )'.format(self.n_elements)