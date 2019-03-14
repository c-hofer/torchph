#ifndef PROFILE //TODO remove this?

#include <torch/extension.h>

#include "calc_pers_cuda.cuh"
#include "vr_comp_cuda.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("CalcPersCuda__find_merge_pairings", &CalcPersCuda::find_merge_pairings, "find_merge_pairings (CUDA)");
    m.def("CalcPersCuda__merge_columns", &CalcPersCuda::merge_columns, "merge_columns (CUDA)");
    m.def("CalcPersCuda__read_barcodes", &CalcPersCuda::read_barcodes, "read_barcodes (CUDA)");
    m.def("CalcPersCuda__calculate_persistence", &CalcPersCuda::calculate_persistence, "calculate_persistence (CUDA)");

    m.def("VRCompCuda__vr_persistence", &VRCompCuda::vr_persistence, "");
    m.def("VRCompCuda__vr_persistence_l1", &VRCompCuda::vr_persistence_l1, "");
    m.def("VRCompCuda__write_combinations_table_to_tensor", &VRCompCuda::write_combinations_table_to_tensor, ""),
    m.def("VRCompCuda__co_faces_from_combinations", &VRCompCuda::co_faces_from_combinations, "");

    m.def("VRCompCuda__l1_norm_distance_matrix", &VRCompCuda::l1_norm_distance_matrix, "");
    m.def("VRCompCuda__l2_norm_distance_matrix", &VRCompCuda::l2_norm_distance_matrix, "");

    pybind11::class_<VRCompCuda::PointCloud2VR>(m, "VRCompCuda__PointCloud2VR")
        .def(pybind11::init<>())
        .def_readwrite("boundary_info_non_vertices", &VRCompCuda::PointCloud2VR::boundary_info_non_vertices)
        .def_readwrite("filtration_values_by_dim", &VRCompCuda::PointCloud2VR::filtration_values_by_dim)
        .def_readwrite("n_simplices_by_dim", &VRCompCuda::PointCloud2VR::n_simplices_by_dim)

        .def_readwrite("simplex_dimension_vector", &VRCompCuda::PointCloud2VR::simplex_dimension_vector)
        .def_readwrite("filtration_values_vector_without_vertices", &VRCompCuda::PointCloud2VR::filtration_values_vector_without_vertices)
        .def_readwrite("filtration_add_eps_hack_values", &VRCompCuda::PointCloud2VR::filtration_add_eps_hack_values)

        .def_readwrite("sort_indices_without_vertices", &VRCompCuda::PointCloud2VR::sort_indices_without_vertices)
        .def_readwrite("sort_indices_without_vertices_inverse", &VRCompCuda::PointCloud2VR::sort_indices_without_vertices_inverse)

        .def_readwrite("sorted_filtration_values_vector", &VRCompCuda::PointCloud2VR::sorted_filtration_values_vector)

        .def_readwrite("boundary_array", &VRCompCuda::PointCloud2VR::boundary_array)
        .def_readwrite("ba_row_i_to_bm_col_i_vector", &VRCompCuda::PointCloud2VR::ba_row_i_to_bm_col_i_vector)

        .def("__call__", &VRCompCuda::PointCloud2VR::operator())

        .def("init_state", &VRCompCuda::PointCloud2VR::init_state, "")
        .def("make_boundary_info_edges", &VRCompCuda::PointCloud2VR::make_boundary_info_edges, "")
        .def("make_boundary_info_non_edges", &VRCompCuda::PointCloud2VR::make_boundary_info_non_edges, "")
        .def("make_simplex_ids_compatible_within_dimensions", &VRCompCuda::PointCloud2VR::make_simplex_ids_compatible_within_dimensions, "")
        .def("make_simplex_dimension_vector", &VRCompCuda::PointCloud2VR::make_simplex_dimension_vector, "")
        .def("make_filtration_values_vector_without_vertices", &VRCompCuda::PointCloud2VR::make_filtration_values_vector_without_vertices, "")
        .def("do_filtration_add_eps_hack", &VRCompCuda::PointCloud2VR::do_filtration_add_eps_hack, "")
        .def("make_sorting_infrastructure", &VRCompCuda::PointCloud2VR::make_sorting_infrastructure, "")
        .def("undo_filtration_add_eps_hack", &VRCompCuda::PointCloud2VR::undo_filtration_add_eps_hack, "")
        .def("make_sorted_filtration_values_vector", &VRCompCuda::PointCloud2VR::make_sorted_filtration_values_vector, "")
        .def("make_boundary_array_rows_unsorted", &VRCompCuda::PointCloud2VR::make_boundary_array_rows_unsorted, "")
        .def("apply_sorting_to_rows", &VRCompCuda::PointCloud2VR::apply_sorting_to_rows, "")
        .def("make_ba_row_i_to_bm_col_i_vector", &VRCompCuda::PointCloud2VR::make_ba_row_i_to_bm_col_i_vector, "")
        // .def("", &VRCompCuda::PointCloud2VR::, "")
        ; 

}

#endif