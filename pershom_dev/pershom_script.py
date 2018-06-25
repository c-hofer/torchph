import os 
import os.path as pth
import pickle
from simplicial_complex import *
from cpu_sorted_boundary_array_implementation import SortedListBoundaryMatrix
import torch 
from time import time
from pershombox import toplex_persistence_diagrams
import chofer_torchex.pershom.pershom_backend as pershom_backend
import yep 
from chofer_torchex.pershom.calculate_persistence import calculate_persistence
import cProfile 
from collections import Counter

# os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)


def test():
    c = None 

    use_cache = False

    if use_cache:
        random_simplicial_complex_path = './random_simplicial_complex.pickle'
        if pth.exists(random_simplicial_complex_path):
            with open(random_simplicial_complex_path, 'br') as f:
                c = pickle.load(f)        
        else:
            c =  random_simplicial_complex(100, 100, 100, 100, 100, 100) 
            with open(random_simplicial_complex_path, 'bw') as f:
                pickle.dump(c, f)
    else:
        c = random_simplicial_complex(100, 100, 100, 100, 100) 

    print('|C| = ', len(c))
    max_red_by_iteration = -1

    # cpu_impl = SortedListBoundaryMatrix(c)
    # cpu_impl.max_pairs = max_red_by_iteration
    bm, col_dim = descending_sorted_boundary_array_from_filtrated_sp(c)

    print(bm[-1]) 

    bm, col_dim = bm.to('cuda'), col_dim.to('cuda')

    
    barcodes_true = toplex_persistence_diagrams(c, list(range(len(c))))
    dgm_true = [Counter(((float(b), float(d)) for b, d in dgm )) for dgm in barcodes_true]


    def my_output_to_dgms(input):
        ret = []
        b, b_e = input    

        for dim, (b_dim, b_dim_e) in enumerate(zip(b, b_e)):
            b_dim, b_dim_e = b_dim.float(), b_dim_e.float()

            tmp = torch.empty_like(b_dim_e)
            tmp.fill_(float('inf'))
            b_dim_e = torch.cat([b_dim_e, tmp], dim=1)


            dgm = torch.cat([b_dim, b_dim_e], dim=0)
            dgm = dgm.tolist()
            dgm = Counter(((float(b), float(d)) for b, d in dgm ))

            ret.append(dgm)

        return ret


    # pr = cProfile.Profile()
    # pr.enable()

    time_start = time()
    output = pershom_backend.calculate_persistence(bm, col_dim, max(col_dim), max_red_by_iteration)
    print(time() - time_start)

    # pr.disable()
    # pr.dump_stats('high_level_profile.cProfile')

    print([[len(x) for x in y] for y in output ])

    dgm_test = my_output_to_dgms(output)

    print('dgm_true lengths:', [len(dgm) for dgm in dgm_true])
    print('dgm_test lengths:',  [len(dgm) for dgm in dgm_test])

    for dgm_test, dgm_true in zip(dgm_test, dgm_true):
        assert(dgm_test == dgm_true)


# x = torch.ones((10,10), device='cuda')
# pershom_backend.__C.my_test_f(x)



    


test()
