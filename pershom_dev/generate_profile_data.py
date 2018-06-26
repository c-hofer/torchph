import sys
import os.path as pth 

import torch 
import numpy as np

from simplicial_complex import *
from cpu_sorted_boundary_array_implementation import SortedListBoundaryMatrix

def generate_profile_input_dump():
    c = random_simplicial_complex(1000, 2000, 4000)
    max_dimension = 2

    bm, simplex_dim = descending_sorted_boundary_array_from_filtrated_sp(c)
    np.savetxt('profiling_pershom/data/boundary_array.np_txt', bm)
    np.savetxt('profiling_pershom/data/boundary_array.np_txt', simplex_dim)

    ind_not_reduced = torch.tensor(list(range(simplex_dim.size(0))))
    ind_not_reduced = ind_not_reduced.masked_select(bm[:, 0] >= 0)

    bm = bm.index_select(0, ind_not_reduced)

    with open('profiling_pershom/data/boundary_array.txt', 'w') as f:
        for row in bm:
            for v in row:
                f.write(str(int(v)))
                f.write(" ")

    with open ('profiling_pershom/data/boundary_array_size.txt', 'w') as f:
        for v in bm.size():        
            f.write(str(int(v)))
            f.write(" ")

    with open ('profiling_pershom/data/ind_not_reduced.txt', 'w') as f:
        for v in ind_not_reduced:        
            f.write(str(int(v)))
            f.write(" ")

    with open ('profiling_pershom/data/column_dimension.txt', 'w') as f:
        for v in simplex_dim:        
            f.write(str(int(v)))
            f.write(" ")

    with open ('profiling_pershom/data/max_dimension.txt', 'w') as f:
        f.write(str(int(max_dimension)))  

    

if __name__ == "__main__":
    generate_profile_input_dump()