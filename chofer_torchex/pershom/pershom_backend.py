import os.path as pth
from torch import Tensor
r"""README

descending_sorted_boundary array: 
    Boundary array which encodes the boundary matrix (BM) for a given filtration in 
    column first order.
    Let BA be the descending_sorted_boundary of BM.

        BA[i, :] -> i-th column of BM. Content encoded as decreasingly sorted list, 
                    embedded into the array with -1 padding from the right. 

        example :
            BA[3, :] = [2, 0, -1, -1] -> boundary(v_3) = v_0 + v_2
            BA[6, :] = [5, 4, 3, -1]  -> boundary(v_6) = v_3 + v_4 + v_5
"""


from torch.utils.cpp_extension import load


__module_file_dir = pth.dirname(pth.realpath(__file__))
__cpp_src_dir = pth.join(__module_file_dir, 'pershom_cpp_src')


# jit compiling the c++ extension
__C = load(
    'pershom_cuda_ext',
    [pth.join(__cpp_src_dir, 'pershom.cpp'),
     pth.join(__cpp_src_dir, 'pershom_cuda.cu')],
    verbose=True)


def find_merge_pairings(
    pivots: Tensor, 
    max_pairs: int
    )->Tensor:
    r"""Finds the pairs which have to be merged in the current iteration. 
    For 
    
    Arguments:
        pivots {Tensor} -- [Nx1: N is the number of columns of the under
        lying descending sorted boundary array]

        max_pairs {int} -- [The output is at most a max_pairs x 2 Tensor]
    
    Returns:
        Tensor -- [Nx2: N is the number of merge pairs]
    """
    return __C.find_merge_pairings(pivots, max_pairs)


def merge_columns_(
    descending_sorted_boundary_array: Tensor, 
    merge_pairs: Tensor
    )->None:
    r"""Executes the given merging operations inplace on the descending 
    sorted boundary array. 
    
    Arguments:
        descending_sorted_boundary_array {Tensor} -- [see readme section top]

        merge_pairs {Tensor} -- [output of a 'find_merge_pairings' call]
    
    Returns:
        None -- []
    """
    __C.merge_columns_(descending_sorted_boundary_array, merge_pairs)


def read_barcodes(
    pivots: Tensor, 
    column_dimension: Tensor, 
    max_dimension: int
    )->[[Tensor], [Tensor]]:
    """Reads the barcodes using the pivot of a reduced boundary array
    
    Arguments:
        pivots {Tensor} -- [pivots is the first column of a 
        descending_sorted_boundary_array]

        column_dimension {Tensor} -- [Vector whose i-th entry is 
        the dimension if the i-th vertex in the given filtration]

        max_dimension {int} -- [dimension of the filtrated simplicial complex]
    
    Returns:
        [[Tensor], [Tensor]] -- [ret[0][i] = non essential barcodes of dimension i
                                 ret[1][i] = birth-times of essential classes]
    """
    return __C.read_barcodes(pivots, column_dimension, max_dimension)


def calculate_persistence(
    descending_sorted_boundary_array: Tensor,
    column_dimension: Tensor,
    max_dimension: int,
    max_pairs: int
    )->[Tensor]:
    """Returns the barcodes of the given encoded boundary array
    
    Arguments:
        descending_sorted_boundary_array {Tensor} -- [see readme section top]

        column_dimension {Tensor} -- [Vector whose i-th entry is 
        the dimension if the i-th vertex in the given filtration]

        max_pairs {int} -- [The output is at most a max_pairs x 2 Tensor]

        max_dimension {int} -- [dimension of the filtrated simplicial complex]
    
    Returns:
        [Tensor] -- [description]
    """
    return __C.calculate_persistence(
        descending_sorted_boundary_array, column_dimension, max_dimension, max_pairs)



    


