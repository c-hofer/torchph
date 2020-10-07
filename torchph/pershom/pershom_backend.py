r"""
This module exposes the C++/CUDA backend functionality for Python.

Terminology
-----------

Descending sorted boundary array:
    Boundary array which encodes the boundary matrix (BM) for a given
    filtration in column first order.
    Let BA be the descending_sorted_boundary of BM, then
    ``BA[i, :]`` is the i-th column of BM.
    Content encoded as decreasingly sorted list, embedded into the array
    with -1 padding from the right.

        Example :
            ``BA[3, :] = [2, 0, -1, -1]``
            then  :math:`\partial(v_3) = v_0 + v_2`

            ``BA[6, :] = [5, 4, 3, -1]``
            then :math:`\partial(v_6) = v_3 + v_4 + v_5`


Compressed descending sorted boundary array:
    Same as *descending sorted boundary array* but rows consisting only of -1
    are omitted.
    This is sometimes used for efficiency purposes and is usually accompanied
    by a vector, ``v``, telling which row of the reduced BA corresponds to
    which row of the uncompressed BA, i.e., ``v[3] = 5`` means that the 3rd
    row of the reduced BA corresponds to the 5th row in the uncompressed
    version.

Birth/Death-time:
    Index of the coresponding birth/death event in the filtration.
    This is always an *integer*.

Birth/Death-value:
    If a filtration is induced by a real-valued function, this corresponds
    to the value of this function corresponding to the birth/death event.
    This is always *real*-valued.

Limitations
-----------

Currently all ``cuda`` backend functionality **only** supports  ``int64_t`` and
``float32_t`` typing.

"""
import warnings
import os.path as pth
from typing import List
from torch import Tensor
from glob import glob


from torch.utils.cpp_extension import load


__module_file_dir = pth.dirname(pth.realpath(__file__))
__cpp_src_dir = pth.join(__module_file_dir, 'pershom_cpp_src')
src_files = []

for extension in ['*.cpp', '*.cu']:
    src_files += glob(pth.join(__cpp_src_dir, extension))

# jit compiling the c++ extension

_failed_compilation_msg = \
    """
    Failed jit compilation in {}.
    Error was `{}`.
    The error will be re-raised calling any function in this module.
    """

__C = None
try:
    __C = load(
        'pershom_cuda_ext',
        src_files,
        verbose=False)

except Exception as ex:
    warnings.warn(_failed_compilation_msg.format(__file__, ex))

    class ErrorThrower(object):
        ex = ex

        def __getattr__(self, name):
            raise self.ex 

    __C = ErrorThrower()


def find_merge_pairings(
        pivots: Tensor,
        max_pairs: int = -1
) -> Tensor:
    """Finds the pairs which have to be merged in the current iteration of the
    matrix reduction.

    Args:
        pivots:
            The pivots of a descending sorted boundary array.
            Expected size is ``Nx1``, where N is the number of columns of the
            underlying descending sorted boundary array.

        max_pairs:
            The output is at most a ``max_pairs x 2`` Tensor. If set to
            default all possible merge pairs are returned.

    Returns:
        The merge pairs, ``p``, for the current iteration of the reduction.
        ``p[i]`` is a merge pair.
        In boundary matrix notation this would mean column ``p[i][0]`` has to
        be merged into column ``p[i][1]``.
    """
    return __C.CalcPersCuda__find_merge_pairings(pivots, max_pairs)


def merge_columns_(
        compr_desc_sort_ba: Tensor,
        merge_pairs: Tensor
) -> None:
    r"""Executes the given merging operations inplace on the descending
    sorted boundary array.

    Args:
        compr_desc_sort_ba:
            see module description top.

        merge_pairs:
            output of a ``find_merge_pairings`` call.

    Returns:
        None
    """
    __C.CalcPersCuda__merge_columns_(compr_desc_sort_ba, merge_pairs)


def read_barcodes(
        pivots: Tensor,
        simplex_dimension: Tensor,
        max_dim_to_read_of_reduced_ba: int
) -> List[List[Tensor]]:
    """Reads the barcodes using the pivot of a reduced boundary array.

    Arguments:
        pivots:
            pivots is the first column of a compr_desc_sort_ba

        simplex_dimension:
            Vector whose i-th entry is the dimension if the i-th simplex in
            the given filtration.

        max_dim_to_read_of_reduced_ba:
            features up to max_dim_to_read_of_reduced_ba are read from the
            reduced boundary array

    Returns:
        List of birth/death times.

        ``ret[0][n]`` are non essential birth/death-times of dimension ``n``.

        ``ret[1][n]`` are birth-times of essential classes of dimension ``n``.
    """
    return __C.CalcPersCuda__read_barcodes(
        pivots,
        simplex_dimension,
        max_dim_to_read_of_reduced_ba)


def calculate_persistence(
        compr_desc_sort_ba: Tensor,
        ba_row_i_to_bm_col_i: Tensor,
        simplex_dimension: Tensor,
        max_dim_to_read_of_reduced_ba: int,
        max_pairs: int = -1
) -> List[List[Tensor]]:
    """Returns the barcodes of the given encoded boundary array.

    Arguments:
        compr_desc_sort_ba:
            A `compressed descending sorted boundary array`,
            see readme section top.

        ba_row_i_to_bm_col_i:
            Vector whose i-th entry is the column index of the boundary
            matrix the i-th row in ``compr_desc_sort_ba corresponds`` to.

        simplex_dimension:
            Vector whose i-th entry is the dimension if the i-th simplex in
            the given filtration

        max_pairs: see ``find_merge_pairings``.

        max_dim_to_read_of_reduced_ba:
            features up to max_dim_to_read_of_reduced_ba are read from the
            reduced boundary array.

    Returns:
        List of birth/death times.

        ``ret[0][n]`` are non essential birth/death-times of dimension ``n``.

        ``ret[1][n]`` are birth-times of essential classes of dimension ``n``.
    """
    return __C.CalcPersCuda__calculate_persistence(
        compr_desc_sort_ba,
        ba_row_i_to_bm_col_i,
        simplex_dimension,
        max_dim_to_read_of_reduced_ba,
        max_pairs)


def vr_persistence_l1(
        point_cloud: Tensor,
        max_dimension: int,
        max_ball_diameter: float = 0.0
) -> List[List[Tensor]]:
    """Returns the barcodes of the Vietoris-Rips complex of a given point cloud
    w.r.t. the l1 (manhatten) distance.

    Args:
        point_cloud:
            Point cloud from which the Vietoris-Rips complex is generated.

        max_dimension:
            The dimension of the used Vietoris-Rips complex.

        max_ball_diameter:
            If not 0, edges whose two defining vertices' distance is greater
            than ``max_ball_diameter`` are ignored.

    Returns:
        List of birth/death times.

        ``ret[0][n]`` are non essential birth/death-*values* of dimension ``n``.

        ``ret[1][n]`` are birth-*values* of essential classes of
        dimension ``n``.
    """
    return __C.VRCompCuda__vr_persistence_l1(
        point_cloud,
        max_dimension,
        max_ball_diameter)


def vr_persistence(
        distance_matrix: Tensor,
        max_dimension: int,
        max_ball_diameter: float = 0.0
) -> List[List[Tensor]]:
    """Returns the barcodes of the Vietoris-Rips complex of a given distance
    matrix.

    **Note**: ``distance_matrix`` is assumed to be a square matrix.
    Practically, symmetry is *not* required and the upper diagonal part is
    *ignored*. For the computation, just the *lower* diagonal part is used.

    Args:
        distance_matrix:
            Distance matrix the Vietoris-Rips complex is based on.

        max_dimension:
            The dimension of the used Vietoris-Rips complex.

        max_ball_diameter:
            If not 0, edges whose two defining vertices' distance is greater
            than ``max_ball_diameter`` are ignored.

    Returns:
        List of birth/death times.

        ``ret[0][n]`` are non essential birth/death-*values* of dimension ``n``.

        ``ret[1][n]`` are birth-*values* of essential classes of
        dimension ``n``.
    """
    return __C.VRCompCuda__vr_persistence(
        distance_matrix,
        max_dimension,
        max_ball_diameter)
