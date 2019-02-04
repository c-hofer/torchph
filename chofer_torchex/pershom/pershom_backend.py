import os.path as pth
from torch import Tensor
from glob import glob
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
src_files = []

for extension in ['*.cpp', '*.cu']:
    src_files += glob(pth.join(__cpp_src_dir, extension))



# jit compiling the c++ extension

try: 
    __C = load(
        'pershom_cuda_ext',
        src_files,
        verbose=True)

except Exception:
    print(" rror in {}. Failed jit compilation. Maybe your CUDA environment is messed up?".format(__file__))

def find_merge_pairings(
    pivots: Tensor, 
    max_pairs: int = -1
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
    return __C.CalcPersCuda__find_merge_pairings(pivots, max_pairs)


def merge_columns_(
    compr_desc_sort_ba: Tensor, 
    merge_pairs: Tensor
    )->None:
    r"""Executes the given merging operations inplace on the descending 
    sorted boundary array. 
    
    Arguments:
        compr_desc_sort_ba {Tensor} -- [see readme section top]

        merge_pairs {Tensor} -- [output of a 'find_merge_pairings' call]
    
    Returns:
        None -- []
    """
    __C.CalcPersCuda__merge_columns_(compr_desc_sort_ba, merge_pairs)


def read_barcodes(
    pivots: Tensor, 
    simplex_dimension: Tensor, 
    max_dim_to_read_of_reduced_ba: int
    )->[[Tensor], [Tensor]]:
    """Reads the barcodes using the pivot of a reduced boundary array
    
    Arguments:
        pivots {Tensor} -- [pivots is the first column of a 
        compr_desc_sort_ba]

        simplex_dimension {Tensor} -- [Vector whose i-th entry is 
        the dimension if the i-th simplex in the given filtration]

        max_dim_to_read_of_reduced_ba {int} -- [features up to max_dim_to_read_of_reduced_ba are 
        read from the reduced boundary array]
    
    Returns:
        [[Tensor], [Tensor]] -- [ret[0][i] = non essential barcodes of dimension i
                                 ret[1][i] = birth-times of essential classes]
    """
    return __C.CalcPersCuda__read_barcodes(pivots, simplex_dimension, max_dim_to_read_of_reduced_ba)


def calculate_persistence(
    compr_desc_sort_ba: Tensor,
    ba_row_i_to_bm_col_i: Tensor, 
    simplex_dimension: Tensor,
    max_dim_to_read_of_reduced_ba: int,
    max_pairs: int = -1
    )->[[Tensor]]:
    """Returns the barcodes of the given encoded boundary array.
    
    Arguments:
        compr_desc_sort_ba {Tensor} -- [see readme section top]

        ba_row_i_to_bm_col_i -- [Vector whose i-th entry is 
        the column index of the boundary matrix the i-th row in 
        compr_desc_sort_ba corresponds to]

        simplex_dimension {Tensor} -- [Vector whose i-th entry is 
        the dimension if the i-th simplex in the given filtration]

        max_pairs {int} -- [see find merge pairings]

        max_dim_to_read_of_reduced_ba {int} -- [features up to max_dim_to_read_of_reduced_ba are 
        read from the reduced boundary array]
    
    Returns:
        [[Tensor], [Tensor]] -- [ret[0][i] = non essential barcodes of dimension i
                                 ret[1][i] = birth-times of essential classes of dimension i]
    """
    return __C.CalcPersCuda__calculate_persistence(
        compr_desc_sort_ba, ba_row_i_to_bm_col_i, simplex_dimension, max_dim_to_read_of_reduced_ba, max_pairs)


def vr_persistence_l1(
    point_cloud: Tensor, 
    max_dimension: int, 
    max_ball_diameter: float=0.0
    )->[[Tensor]]:
    """Returns the barcodes of the Vietoris-Rips complex of a given point cloud.
    
    Args:
        point_cloud (Tensor): Point cloud from which the Vietoris-Rips complex is generated.
        max_dimension (int): The dimension of the used Vietoris-Rips complex. 
        max_ball_diameter (float): If not 0, edges whose two defining vertices' distance is greater than max_ball_diameter are ignored. 
    
    Returns:
        [[Tensor], [Tensor]] : [ret[0][i] = non essential barcodes of dimension i
                                ret[1][i] = birth-times of essential classes of dimension i]
    """
    return __C.VRCompCuda__vr_persistence(point_cloud, max_dimension, max_ball_diameter, 'l1')






    


