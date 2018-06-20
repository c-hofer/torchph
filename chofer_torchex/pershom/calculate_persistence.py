from .pershom_backend import find_merge_pairings, merge_columns_, read_barcodes
import torch 
from torch import Tensor

def calculate_persistence(
    descending_sorted_boundary_array: Tensor,
    column_dimension: Tensor,
    max_dimension: int,
    max_pairs: int
    )->[Tensor]:

    iterations = 0
    pivots = None
    while True:
        pivots = descending_sorted_boundary_array[:, 0].unsqueeze(1).contiguous()

        try:
            merge_pairings = find_merge_pairings(pivots, max_pairs)

        except Exception as ex:
            print(ex)

            print("Reached end of reduction after {} iterations".format(iterations))
            break

        merge_columns_(descending_sorted_boundary_array, merge_pairings)
        iterations += 1

    barcodes = read_barcodes(pivots, column_dimension, max_dimension)

    return barcodes 
        
