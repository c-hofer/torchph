import os 
import os.path as pth
import pickle
from simplicial_complex import *
from cpu_sorted_boundary_array_implementation import SortedListBoundaryMatrix
import torch 
from pershombox import toplex_persistence_diagrams
from collections import Counter


def generate():    
    
    output_path = "/tmp/random_simplicial_complexes"
    try: 
        os.mkdir(output_path)
    except:
        pass
    
    args = [(100, 200, 300), 
            (100, 200, 300, 400), 
            (100, 100, 100, 100, 100)]


    for arg in args: 
        c = random_simplicial_complex(*arg)
        file_name_stub = os.path.join(output_path, "random_sp__args__" + "_".join(str(x) for x in arg))
    
        bm, col_dim = descending_sorted_boundary_array_from_filtrated_sp(c)
        bm, col_dim = bm.to('cuda'), col_dim.to('cuda')
        
        barcodes_true = toplex_persistence_diagrams(c, list(range(len(c))))
        dgm_true = [Counter(((float(b), float(d)) for b, d in dgm )) for dgm in barcodes_true]

        with open(file_name_stub + ".pickle", 'wb') as f:
            pickle.dump({'calculate_persistence_args': (bm, col_dim, max(col_dim)), 
                         'expected_result': dgm_true}, 
                         f)
           


if __name__ == "__main__":
    generate()
