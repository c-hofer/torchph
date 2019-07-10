import pytest
import torch
import chofer_torchex.pershom.pershom_backend as pershom_backend
import glob
import os
import numpy

from itertools import combinations
from scipy.special import binom
from itertools import combinations
from collections import Counter


__C = pershom_backend.__C
EPSILON = 0.0001


def test_l2_distance_matrix():
    point_cloud = [(0, 0), (1, 0), (0, 0.5), (1, 1.5)]
    point_cloud = torch.tensor(point_cloud, device='cuda', dtype=torch.float)

    D = __C.VRCompCuda__l2_norm_distance_matrix(point_cloud)

    for i, j in combinations(range(point_cloud.size(0)), 2):
        l2_dist = (point_cloud[i] - point_cloud[j]).pow(2).sum().sqrt()
        l2_dist = float(l2_dist)

        assert float(D[i, j]) == l2_dist


def test_l1_distance_matrix():
    point_cloud = [(0, 0), (1, 0), (0, 0.5), (1, 1.5)]
    point_cloud = torch.tensor(point_cloud, device='cuda', dtype=torch.float)

    D = __C.VRCompCuda__l1_norm_distance_matrix(point_cloud)

    for i, j in combinations(range(point_cloud.size(0)), 2):
        l1_dist = (point_cloud[i] - point_cloud[j]).abs().sum()
        l1_dist = float(l1_dist)

        assert float(D[i, j]) == l1_dist


def test_co_faces_from_combinations__ground_truth():
    faces = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]
    comb = [sorted(cf, reverse=True) for cf in combinations(range(len(faces)), 3)]

    faces_t = torch.tensor(faces, device='cuda', dtype=torch.int64)
    comb_t = torch.tensor(comb, device='cuda', dtype=torch.int64)

    result = __C.VRCompCuda__co_faces_from_combinations(comb_t, faces_t)

    expected_result = [[2, 1, 0], [4, 3, 0]]

    assert result.tolist() == expected_result


def test_co_faces_from_combinations__correct_size_if_just_one_co_face():
    faces = [(1, 0), (2, 0), (2, 1), (3, 0)]
    comb = [sorted(cf, reverse=True) for cf in combinations(range(len(faces)), 3)]

    faces_t = torch.tensor(faces, device='cuda', dtype=torch.int64)
    comb_t = torch.tensor(comb, device='cuda', dtype=torch.int64)

    result = __C.VRCompCuda__co_faces_from_combinations(comb_t, faces_t)

    assert result.dim() == 2
    assert result.size(0) == 1
    
    expected_result = [[2, 1, 0]]

    assert result.tolist() == expected_result


def test_co_faces_from_combinations__no_co_face():
    faces = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    comb = [sorted(cf, reverse=True) for cf in combinations(range(len(faces)), 3)]

    faces_t = torch.tensor(faces, device='cuda', dtype=torch.int64)
    comb_t = torch.tensor(comb, device='cuda', dtype=torch.int64)

    result = __C.VRCompCuda__co_faces_from_combinations(comb_t, faces_t)

    expected_result = []

    assert result.tolist() == expected_result


@pytest.mark.parametrize("n_vertices, dim_faces", [(10, 2), (8, 3)])
def test_co_faces_from_combinations__result(n_vertices, dim_faces):
    faces = [sorted(f) for f in combinations(range(n_vertices), dim_faces)]
    comb = [sorted(cf) for cf in combinations(range(len(faces)), dim_faces + 1)]

    faces_t = torch.tensor(faces, device='cuda', dtype=torch.int64)
    comb_t = torch.tensor(comb, device='cuda', dtype=torch.int64)

    expected_result = []
    for c in comb:
        tmp = sum([faces[i] for i in c], [])
        tmp = Counter(tmp)

        if all(v == 2 for v in tmp.values()):
            expected_result.append(c)

    expected_result = {tuple(x) for x in expected_result}

    result = __C.VRCompCuda__co_faces_from_combinations(comb_t, faces_t)

    result = {tuple(x) for x in result.tolist()}

    assert expected_result == result


@pytest.mark.parametrize(
    "max_dimension, max_ball_radius",
    [(2, 0.0), (2, 1.0), (2, 2.0), (3, 0.0), (3, 1.0), (3, 2.0)])
def test_VietorisRipsArgsGenerator__l1_excessive_state_testing_of_all_intermediate_steps(max_dimension, max_ball_radius):
    # ASSERTS that the number of edges is > 0!!!
    point_cloud = [(0, 0), (1, 0), (0, 0.5), (1, 1.5)]
    point_cloud = torch.tensor(point_cloud, device='cuda', dtype=torch.float)
    distance_matrix = pershom_backend.__C.VRCompCuda__l1_norm_distance_matrix(point_cloud)

    def l1_norm(x, y):
        return float((x-y).abs().sum())

    testee = pershom_backend.__C.VRCompCuda__VietorisRipsArgsGenerator()
    #
    # Test: init_state
    #
    testee.init_state(distance_matrix, max_dimension, max_ball_radius)

    assert len(testee.filtration_values_by_dim) == 1
    assert testee.n_simplices_by_dim[0] == point_cloud.size(0)
    assert testee.filtration_values_by_dim[0].size(0) == point_cloud.size(0)
    assert all(testee.filtration_values_by_dim[0] == 0)

    #
    # Test: make_boundary_info_edges
    #
    testee.make_boundary_info_edges()

    assert len(testee.filtration_values_by_dim) == 2

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

    expected_boundaries = torch.tensor(
        expected_boundaries,
        device='cuda',
        dtype=torch.int64)
    if expected_boundaries.numel() == 0:
        expected_boundaries = torch.empty(
            (0, 2),
            device='cuda',
            dtype=torch.int64)

    expected_filtration_values = torch.tensor(
        expected_filtration_values,
        device='cuda')

    assert testee.n_simplices_by_dim[1] == expected_n_dim_1_simplices
    assert testee.boundary_info_non_vertices[0].size(0) == testee.filtration_values_by_dim[1].size(0)
    assert testee.boundary_info_non_vertices[0].equal(expected_boundaries)
    assert testee.filtration_values_by_dim[1].equal(expected_filtration_values)
    assert testee.filtration_values_by_dim[1].dim() == 1 

    #
    # Test: make_boundary_info_non_edges
    #
    testee.make_boundary_info_non_edges()

    for dim in range(2, max_dimension + 1):
        n_dim_min_one_simplices = testee.n_simplices_by_dim[dim-1]
        dim_min_one_filtration_values = testee.filtration_values_by_dim[dim-1]
        dim_min_one_boundary_info = testee.boundary_info_non_vertices[dim-2].tolist()
        expected_n_simplices = binom(n_dim_min_one_simplices, dim + 1)

        expected_boundaries = []
        expected_filtration_values = []

        for boundaries in combinations(range(n_dim_min_one_simplices), dim + 1):

            boundaries = list(boundaries)
            boundaries_of_boundaries = sum((dim_min_one_boundary_info[i] for i in boundaries), [])
            c = Counter(boundaries_of_boundaries) 
            if all(v == 2 for v in c.values()):            
                expected_boundaries.append(tuple(sorted(boundaries, reverse=True)))
        
        expected_boundaries = sorted(expected_boundaries)
        
        for boundaries in expected_boundaries:
            expected_filtration_values.append(
                max(dim_min_one_filtration_values[b] for b in boundaries) 
            )

        expected_filtration_values = torch.tensor(
            expected_filtration_values,
            device='cuda')

        assert expected_boundaries == [tuple(x) for x in testee.boundary_info_non_vertices[dim-1].tolist()]
        assert expected_filtration_values.equal(testee.filtration_values_by_dim[dim])
        assert testee.n_simplices_by_dim[dim] == len(expected_boundaries)

    #
    # Test: make_simplex_ids_compatible_within_dimensions
    #
    testee.make_simplex_ids_compatible_within_dimensions()
    for dim, boundary_info in enumerate(testee.boundary_info_non_vertices[1:], start=2):
        if boundary_info.numel() > 0:
            min_id = sum(testee.n_simplices_by_dim[:dim-1])
            max_id = min_id + testee.n_simplices_by_dim[dim-1]

            simp_ids = boundary_info.view(-1).tolist()

            assert min_id <= min(simp_ids)
            assert max_id >= max(simp_ids)
        else:
            assert testee.n_simplices_by_dim[dim] == 0   

    #
    # Test: make_simplex_dimension_vector
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

    #
    # Test: make_filtration_values_vector_without_vertices
    #
    testee.make_filtration_values_vector_without_vertices()
    assert testee.filtration_values_vector_without_vertices.dim() == 1
    assert testee.filtration_values_vector_without_vertices.size(0) == sum(testee.n_simplices_by_dim[1:])

    #
    # Test: do_filtration_add_eps_hack
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

    #
    # Test: make_sorting_infrastructure
    #
    testee.make_sorting_infrastructure()
    assert testee.sort_indices_without_vertices.dim() == 1
    assert testee.sort_indices_without_vertices.size(0) == sum(testee.n_simplices_by_dim[1:])

    tmp = testee.sort_indices_without_vertices_inverse.index_select(
        0, testee.sort_indices_without_vertices)
    expected_tmp = torch.tensor(list(range(sum(testee.n_simplices_by_dim[1:]))), device='cuda')
    assert tmp.equal(expected_tmp)

    # Test 
    # undo_filtration_add_eps_hack
    #
    testee.undo_filtration_add_eps_hack()

    tmp = testee.n_simplices_by_dim
    for i, boundary_info in enumerate(testee.boundary_info_non_vertices):
        if i == 0: continue

        id_offset = sum(tmp[1:i+1])

        for j, boundaries in enumerate(boundary_info.tolist()):
            id = id_offset + j

            filt_b = []
            
            for b in boundaries:
                filt_b.append(
                    testee.filtration_values_vector_without_vertices[int(b) - point_cloud.size(0)])
            assert testee.filtration_values_vector_without_vertices[id] == max(filt_b) 

    #
    # Test: make_sorted_filtration_values_vector
    #
    testee.make_sorted_filtration_values_vector()
    assert testee.sorted_filtration_values_vector.equal(
        testee.sorted_filtration_values_vector.sort(0)[0]
    )

    #
    # Test: make_boundary_array_rows_unsorted
    #
    testee.make_boundary_array_rows_unsorted()

    assert testee.boundary_array.size() \
        == (sum(testee.n_simplices_by_dim[1:]), 4 if max_dimension == 0 else 2*(max_dimension +1))

    #
    # Test: apply_sorting_to_rows
    #
    testee.apply_sorting_to_rows()
    for i, row in enumerate(testee.boundary_array.tolist()):
        simplex_id = i + testee.n_simplices_by_dim[0]
        row = [x for x in row if x != -1]
        simplex_dim = len(row) - 1
        assert simplex_dim == testee.simplex_dimension_vector[simplex_id]
        b_filt_vals = []
        for b in row:
            assert testee.simplex_dimension_vector[b] == simplex_dim - 1
            b_filt_vals.append(float(testee.sorted_filtration_values_vector[b]))


        assert sorted(b_filt_vals, reverse=True) == b_filt_vals

        if simplex_dim > 1:
            assert float(testee.sorted_filtration_values_vector[simplex_id]) == max(b_filt_vals)

    #
    # Test: make_ba_row_i_to_bm_col_i_vector
    #
    testee.make_ba_row_i_to_bm_col_i_vector()
    assert testee.ba_row_i_to_bm_col_i_vector.dim() == 1
    assert testee.ba_row_i_to_bm_col_i_vector.size(0) == sum(testee.n_simplices_by_dim[1:])

    expected_ba_row_i_to_bm_col_i_vector = range(testee.n_simplices_by_dim[0], sum(testee.n_simplices_by_dim))
    expected_ba_row_i_to_bm_col_i_vector = list(expected_ba_row_i_to_bm_col_i_vector)
    expected_ba_row_i_to_bm_col_i_vector = torch.tensor(expected_ba_row_i_to_bm_col_i_vector, device='cuda')
    assert testee.ba_row_i_to_bm_col_i_vector.equal(expected_ba_row_i_to_bm_col_i_vector)


random_point_clouds = []
for pth in glob.glob(os.path.join(os.path.dirname(__file__), "test_pershom_backend_data/random_point_clouds/*.txt")):
    X = numpy.loadtxt(pth).tolist()
    X = torch.tensor(X, device='cuda', dtype=torch.float)
    random_point_clouds.append(X)


@pytest.mark.parametrize("test_args", random_point_clouds)
def test_VietorisRipsArgsGenerator__l1_valid_calculate_persistence_args(test_args):
    point_cloud = test_args
    point_cloud = point_cloud.float().to('cuda')
    distance_matrix = pershom_backend.__C.VRCompCuda__l1_norm_distance_matrix(point_cloud)
    max_dimension = 2
    max_ball_radius = 0

    testee = pershom_backend.__C.VRCompCuda__VietorisRipsArgsGenerator()
    ba, ba_row_i_to_bm_col_i, simplex_dimension, sorted_filtration_values_vector = testee(
        distance_matrix,
        max_dimension,
        max_ball_radius)

    if ba.size(0) > 0:

        assert ba.dim() == 2

        assert ba_row_i_to_bm_col_i.dim() == 1
        assert ba.size(0) == ba_row_i_to_bm_col_i.size(0)
        n_simplices = point_cloud.size(0) + ba.size(0)

        assert simplex_dimension.dim() == 1
        assert simplex_dimension.size(0) == n_simplices

        assert sorted_filtration_values_vector.dim() == 1
        assert sorted_filtration_values_vector.size(0) == n_simplices

        # discard right half of columns, which are filled with -1 ...
        content_column_border = (max_dimension if max_dimension != 0 else 1) + 1
        ba_right = ba[:, content_column_border:].contiguous()
        ba_right = ba_right.view(ba_right.numel())
        ba = ba[:, :content_column_border].tolist()
        # ba = [[x for x in row if x != -1] for row in ba]

        assert all(x == -1 for x in ba_right) 

        simplex_dimension = simplex_dimension.tolist()
        assert max(simplex_dimension) == max_dimension

        ba_row_i_to_bm_col_i = ba_row_i_to_bm_col_i.tolist()
        sorted_filtration_values_vector = sorted_filtration_values_vector.tolist()

        for i, row_i in enumerate(ba):
            simplex_id = i + point_cloud.size(0)
            simplex_dim = simplex_dimension[simplex_id]
            simplex_filt_val = sorted_filtration_values_vector[simplex_id]        

            # sorted descendingly by id?
            assert row_i == sorted(row_i, reverse=True)
            boundary_ids = [x for x in row_i if x != -1]

            # dim n simplex has n+1 boundaries?
            assert len(boundary_ids) == simplex_dim + 1
            boundary_filt_vals = [sorted_filtration_values_vector[x] for x in boundary_ids]

            # boundary filtration values are coherent with ids?
            assert boundary_filt_vals == sorted(boundary_filt_vals, reverse=True)

            # simplex_filt_val is maximum of boundary filtration values?
            if simplex_dim > 1 : 
                assert abs(max(boundary_filt_vals) - simplex_filt_val) < EPSILON

            # boundary of simplex has dim - 1 ?
            boundary_dims = [simplex_dimension[b_id] for b_id in boundary_ids]
            assert all(list(dim == simplex_dim - 1 for dim in boundary_dims))

            if simplex_dim > 1:
                boundary_of_boundaries = sum([ba[b_id - point_cloud.size(0)] for b_id in boundary_ids], [])
                boundary_of_boundaries = [x for x in boundary_of_boundaries if x != -1]

                c = Counter(boundary_of_boundaries)

                assert all(list(v == 2 for v in c.values()))
    # ba.size(0) == 0
    else:

        assert ba_row_i_to_bm_col_i.numel() == 0
        assert simplex_dimension.tolist() == [0]*point_cloud.size(0)
        assert sorted_filtration_values_vector.tolist() == [0]*point_cloud.size(0)
