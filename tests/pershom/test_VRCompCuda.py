import pytest
import torch
import chofer_torchex.pershom.pershom_backend as pershom_backend

from scipy.special import binom
from itertools import combinations

__C = pershom_backend.__C


    

def test_vr_l1_generate_calculate_persistence_args__ground_truth_1():

    point_cloud = [(0, 0), (1, 0), (0, 0.5)]
    point_cloud = torch.tensor(point_cloud, device='cuda', dtype=torch.float)

    ba_expected = torch.tensor(
        [[ 2,  0, -1, -1, -1, -1],
        [ 1,  0, -1, -1, -1, -1],
        [ 2,  1, -1, -1, -1, -1],
        [ 5,  4,  3, -1, -1, -1]], device='cuda', dtype=torch.int64
    )

    ba_row_i_to_bm_col_i_expected = torch.tensor([3, 4, 5, 6], device='cuda', dtype=torch.int64)

    simplex_dimension_expected = torch.tensor([0, 0, 0, 1, 1, 1, 2], device='cuda', dtype=torch.int64)

    sorted_filtration_values_vector_expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 1.5], device='cuda', dtype=torch.float)

    
    args = __C.VRCompCuda__vr_l1_generate_calculate_persistence_args(
        point_cloud,2, 0)

    ba, ba_row_i_to_bm_col_i, simplex_dimension, sorted_filtration_values_vector = args

    assert ba.equal(ba_expected)
    assert ba_row_i_to_bm_col_i.equal(ba_row_i_to_bm_col_i_expected)
    assert simplex_dimension.equal(simplex_dimension_expected)
    assert sorted_filtration_values_vector.equal(sorted_filtration_values_vector_expected)


def test_vr_l1_generate_calculate_persistence_args__sanity():
    point_cloud = [(0, 0), (1, 0), (0, 0.5), (1, 1.5)]
    point_cloud = torch.tensor(point_cloud, device='cuda', dtype=torch.float)

    max_dimension = 3

    args = __C.VRCompCuda__vr_l1_generate_calculate_persistence_args(
        point_cloud, max_dimension, 0)

    ba, ba_row_i_to_bm_col_i, simplex_dimension, sorted_filtration_values_vector = args

    n_0 = 4
    n_1 = binom(n_0, 2)
    n_2 = binom(n_1, 3)
    n_3 = binom(n_2, 4) 

    # check dimensions of output ...
    assert ba.size(0) == n_1 + n_2 + n_3 
    assert ba.size(1) == 2*(max_dimension+1)
    assert ba_row_i_to_bm_col_i.size(0) == ba.size(0)
    assert simplex_dimension.size(0) == ba.size(0) + n_0
    assert sorted_filtration_values_vector.size(0) == ba.size(0) + n_0


    # sanity check ...
    ba = ba[:, :max_dimension + 1].tolist()
    simplex_dimension = simplex_dimension.tolist()
    sorted_filtration_values_vector = sorted_filtration_values_vector.tolist()

    for i, row_i in enumerate(ba):
        cycle_id = i + n_0
        cycle_dim = simplex_dimension[cycle_id]
        cycle_filt_val = sorted_filtration_values_vector[cycle_id]

        for boundary_id in row_i: 
            if boundary_id == -1: continue

            assert cycle_dim - 1 == simplex_dimension[boundary_id]
            assert cycle_filt_val >= sorted_filtration_values_vector[boundary_id]


@pytest.mark.parametrize("max_ball_radius", [0, 0.5, 1.0, 1.5, 2.0, 2.5])  
def test_vr_l1_generate_calculate_persistence_args__max_ball_radius(max_ball_radius):
    point_cloud = [(0, 0), (1, 0), (0, 0.5), (1, 1.5)]
    point_cloud = torch.tensor(point_cloud, device='cuda', dtype=torch.float)

    max_dimension = 1

    n_0 = point_cloud.size(0)

    if max_ball_radius == 0:

        n_1 = binom(n_0, 2)

    else:   
        n_1 = 0
        for i, j in combinations(range(point_cloud.size(0)), 2):
            x = point_cloud[i]
            y = point_cloud[j]
            d = (x-y).norm(p=1)

            if d <= max_ball_radius: n_1 += 1

    n_2 = binom(n_1, 3)
    
    args = __C.VRCompCuda__vr_l1_generate_calculate_persistence_args(
        point_cloud, max_dimension, max_ball_radius)

    ba, ba_row_i_to_bm_col_i, simplex_dimension, sorted_filtration_values_vector = args

    assert ba.size(0) + n_0 == n_0 + n_1 


    











