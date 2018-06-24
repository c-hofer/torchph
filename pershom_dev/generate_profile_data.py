import sys
import os.path as pth 

import torch 
import numpy as np

from chofer_torchex.pershom.calculate_persistence import calculate_persistence
from simplicial_complex import *
from cpu_sorted_boundary_array_implementation import SortedListBoundaryMatrix

def generate_test_input_dump():
    c = random_simplicial_complex(1000, 2000, 4000)
    max_dimension = 2

    bm, col_dim = descending_sorted_boundary_array_from_filtrated_sp(c)

    with open('profiling_pershom/data/boundary_array.txt', 'w') as f:
        for row in bm:
            for v in row:
                f.write(str(int(v)))
                f.write(" ")

    with open ('profiling_pershom/data/boundary_array_size.txt', 'w') as f:
        for v in bm.size():        
            f.write(str(int(v)))
            f.write(" ")

    with open ('profiling_pershom/data/column_dimension.txt', 'w') as f:
        for v in col_dim:        
            f.write(str(int(v)))
            f.write(" ")

    with open ('profiling_pershom/data/max_dimension.txt', 'w') as f:
        f.write(str(int(max_dimension)))  

    np.savetxt('profiling_pershom/data/boundary_array.np_txt', bm)
    np.savetxt('profiling_pershom/data/boundary_array.np_txt', col_dim)
    

if __name__ == "__main__":
    generate_test_input_dump()