import os 
import os.path as pth
import pickle
from simplicial_complex import *
from cpu_sorted_boundary_array_implementation import SortedListBoundaryMatrix
import torch 
from time import time
# from pershombox import toplex_persistence_diagrams
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
    
    ind_not_reduced = torch.tensor(list(range(col_dim.size(0)))).to('cuda').detach()
    ind_not_reduced = ind_not_reduced.masked_select(bm[:, 0] >= 0).long().detach()
    bm = bm.index_select(0, ind_not_reduced).detach()

    yep.start('profiling_pershom/profile.google-pprof')
    
    for i in range(10):
        time_start = time()
        output = pershom_backend.calculate_persistence(bm.clone(), ind_not_reduced.clone(), col_dim.clone(), max(col_dim))
        print(time() - time_start)
    yep.stop()

    # pr.disable()
    # pr.dump_stats('high_level_profile.cProfile')

    print([[len(x) for x in y] for y in output ])

    dgm_test = my_output_to_dgms(output)

    print('dgm_true lengths:', [len(dgm) for dgm in dgm_true])
    print('dgm_test lengths:',  [len(dgm) for dgm in dgm_test])

    for dgm_test, dgm_true in zip(dgm_test, dgm_true):
        assert(dgm_test == dgm_true)


def vr_l1_persistence_performance_test():
    pc = torch.randn(20,3)
    pc = torch.tensor(pc, device='cuda', dtype=torch.float)

    max_dimension = 2
    max_ball_radius = 0 

    yep.start('profiling_pershom/profile.google-pprof')
    time_start = time()
    res = pershom_backend.__C.VRCompCuda__vr_persistence(pc, max_dimension, max_ball_radius, 'l1')
    print(time() - time_start)
    yep.stop()
    print(res[0][0])



    
vr_l1_persistence_performance_test()
