import chofer_torchex.pershom.pershom_backend as pershom_backend
import torch
import time
from scipy.special import binom
from itertools import combinations

# get_distance_matrix = pershom_backend.__C.VRCompCuda__l1_norm_distance_matrix

def l1_norm(x, y):
    return float((x-y).abs().sum())

testee = pershom_backend.__C.VRCompCuda__PointCloud2VR_factory("l1")

point_cloud = torch.rand((10, 3), device='cuda')
max_dimension = 2
max_ball_radius = 0 


# Test
# init_state 
#
testee.init_state(point_cloud, max_dimension, max_ball_radius)

assert len(testee.filtration_values_by_dim) == 1
assert testee.n_simplices_by_dim[0] == point_cloud.size(0)
assert testee.filtration_values_by_dim[0].size(0) == point_cloud.size(0)
assert all(testee.filtration_values_by_dim[0] == 0)


# Test 
# make_boundary_info_edges
#
testee.make_boundary_info_edges()

assert len(testee.filtration_values_by_dim) == 2
assert testee.n_simplices_by_dim[1] == binom(testee.n_simplices_by_dim[0], 2) 

expected_n_dim_1_simplices = 0 
expected_filtration_values = []
expected_boundaries = []
threshold = max_ball_radius if max_ball_radius > 0 else float('inf')
for i, j in combinations(range(point_cloud.size(0)), 2):
    
    norm = l1_norm(point_cloud[i], point_cloud[j])
    
    if norm <= threshold:
        expected_n_dim_1_simplices += 1
        expected_boundaries.append(sorted((i, j), reverse=True))

expected_boundaries = sorted(expected_boundaries)

for i, j in expected_boundaries:
    norm = l1_norm(point_cloud[i], point_cloud[j])
    expected_filtration_values.append(norm)

expected_boundaries = torch.tensor(expected_boundaries, device='cuda')
expected_filtration_values = torch.tensor(expected_filtration_values, device='cuda')

assert testee.n_simplices_by_dim[1] == expected_n_dim_1_simplices
assert testee.boundary_info_non_vertices[0].size(0) == testee.filtration_values_by_dim[1].size(0)
assert testee.boundary_info_non_vertices[0].equal(expected_boundaries)
assert testee.filtration_values_by_dim[1].equal(expected_filtration_values)
assert testee.filtration_values_by_dim[1].dim() == 1 


# Test
# make_boundary_info_non_edges
#
testee.make_boundary_info_non_edges()

for dim in range(2, max_dimension + 1):
    n_dim_min_one_simplices = testee.n_simplices_by_dim[dim-1]
    dim_min_one_filtration_values = testee.filtration_values_by_dim[dim-1]
    expected_n_simplices = binom(n_dim_min_one_simplices, dim + 1)

    expected_boundaries = []
    expected_filtration_values = []

    for boundaries in combinations(range(n_dim_min_one_simplices), dim + 1):
        expected_boundaries.append(sorted(tuple(boundaries), reverse=True))
    
    expected_boundaries = sorted(expected_boundaries)
    
    for boundaries in expected_boundaries:
        expected_filtration_values.append(
            max(dim_min_one_filtration_values[b] for b in boundaries) 
        )

    expected_boundaries = torch.tensor(expected_boundaries, device='cuda')
    expected_filtration_values = torch.tensor(expected_filtration_values, device='cuda')

    assert expected_boundaries.equal(testee.boundary_info_non_vertices[dim-1])
    assert expected_filtration_values.equal(testee.filtration_values_by_dim[dim])
    assert testee.n_simplices_by_dim[dim] == expected_boundaries.size(0)


# Test
# make_simplex_ids_compatible_within_dimensions
#
testee.make_simplex_ids_compatible_within_dimensions()

tmp = [0] + testee.n_simplices_by_dim
for i, boundary_info in enumerate(testee.boundary_info_non_vertices):
    expected_simplex_ids = set(range(sum(tmp[:i+1]), sum(tmp[:i+2])))
    simplex_ids = set(boundary_info.view(-1).tolist())

    assert simplex_ids == expected_simplex_ids


# Test
# make_simplex_dimension_vector
#
testee.make_simplex_dimension_vector()

tmp = testee.n_simplices_by_dim
for i in range(max_dimension+1):
    a = sum(tmp[:i])
    b = sum(tmp[:i+1])
    
    slice_i  = testee.simplex_dimension_vector[a:b]
    assert  slice_i.numel() == 0 or (slice_i.equal(torch.empty_like(slice_i).fill_(i)))

assert testee.simplex_dimension_vector.dim() == 1
assert testee.simplex_dimension_vector.size(0) == sum(tmp) 


# Test
# make_filtration_values_vector_without_vertices
#
testee.make_filtration_values_vector_without_vertices()
assert testee.filtration_values_vector_without_vertices.dim() == 1
assert testee.filtration_values_vector_without_vertices.size(0) == sum(testee.n_simplices_by_dim[1:])


# Test 
# do_filtration_add_eps_hack
#
testee.do_filtration_add_eps_hack()

tmp = testee.n_simplices_by_dim
for i, boundary_info in enumerate(testee.boundary_info_non_vertices):
    if i == 0: continue 

    id_offset = sum(tmp[1:i+1])


    for j, boundaries in enumerate(boundary_info.tolist()):
        id = id_offset + j

        for b in boundaries:
            assert testee.filtration_values_vector_without_vertices[int(b) - point_cloud.size(0)] <= \
            testee.filtration_values_vector_without_vertices[id]  



# print(c(torch.rand((10, 3), device='cuda'), 2, 0))
# print(c.filtration_values_by_dim)