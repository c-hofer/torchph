#ifndef PROFILE //TODO remove this?

#include <torch/extension.h>

#include "calc_pers_cuda.cuh"
#include "vr_comp_cuda.cuh"
#include "vertex_filtration_comp_cuda.h"

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

    pybind11::class_<VRCompCuda::VietorisRipsArgsGenerator>(m, "VRCompCuda__VietorisRipsArgsGenerator")
        .def(pybind11::init<>())
        .def_readwrite("boundary_info_non_vertices", &VRCompCuda::VietorisRipsArgsGenerator::boundary_info_non_vertices)
        .def_readwrite("filtration_values_by_dim", &VRCompCuda::VietorisRipsArgsGenerator::filtration_values_by_dim)
        .def_readwrite("n_simplices_by_dim", &VRCompCuda::VietorisRipsArgsGenerator::n_simplices_by_dim)

        .def_readwrite("simplex_dimension_vector", &VRCompCuda::VietorisRipsArgsGenerator::simplex_dimension_vector)
        .def_readwrite("filtration_values_vector_without_vertices", &VRCompCuda::VietorisRipsArgsGenerator::filtration_values_vector_without_vertices)
        .def_readwrite("filtration_add_eps_hack_values", &VRCompCuda::VietorisRipsArgsGenerator::filtration_add_eps_hack_values)

        .def_readwrite("sort_indices_without_vertices", &VRCompCuda::VietorisRipsArgsGenerator::sort_indices_without_vertices)
        .def_readwrite("sort_indices_without_vertices_inverse", &VRCompCuda::VietorisRipsArgsGenerator::sort_indices_without_vertices_inverse)

        .def_readwrite("sorted_filtration_values_vector", &VRCompCuda::VietorisRipsArgsGenerator::sorted_filtration_values_vector)

        .def_readwrite("boundary_array", &VRCompCuda::VietorisRipsArgsGenerator::boundary_array)
        .def_readwrite("ba_row_i_to_bm_col_i_vector", &VRCompCuda::VietorisRipsArgsGenerator::ba_row_i_to_bm_col_i_vector)

        .def("__call__", &VRCompCuda::VietorisRipsArgsGenerator::operator())

        .def("init_state", &VRCompCuda::VietorisRipsArgsGenerator::init_state, "")
        .def("make_boundary_info_edges", &VRCompCuda::VietorisRipsArgsGenerator::make_boundary_info_edges, "")
        .def("make_boundary_info_non_edges", &VRCompCuda::VietorisRipsArgsGenerator::make_boundary_info_non_edges, "")
        .def("make_simplex_ids_compatible_within_dimensions", &VRCompCuda::VietorisRipsArgsGenerator::make_simplex_ids_compatible_within_dimensions, "")
        .def("make_simplex_dimension_vector", &VRCompCuda::VietorisRipsArgsGenerator::make_simplex_dimension_vector, "")
        .def("make_filtration_values_vector_without_vertices", &VRCompCuda::VietorisRipsArgsGenerator::make_filtration_values_vector_without_vertices, "")
        .def("do_filtration_add_eps_hack", &VRCompCuda::VietorisRipsArgsGenerator::do_filtration_add_eps_hack, "")
        .def("make_sorting_infrastructure", &VRCompCuda::VietorisRipsArgsGenerator::make_sorting_infrastructure, "")
        .def("undo_filtration_add_eps_hack", &VRCompCuda::VietorisRipsArgsGenerator::undo_filtration_add_eps_hack, "")
        .def("make_sorted_filtration_values_vector", &VRCompCuda::VietorisRipsArgsGenerator::make_sorted_filtration_values_vector, "")
        .def("make_boundary_array_rows_unsorted", &VRCompCuda::VietorisRipsArgsGenerator::make_boundary_array_rows_unsorted, "")
        .def("apply_sorting_to_rows", &VRCompCuda::VietorisRipsArgsGenerator::apply_sorting_to_rows, "")
        .def("make_ba_row_i_to_bm_col_i_vector", &VRCompCuda::VietorisRipsArgsGenerator::make_ba_row_i_to_bm_col_i_vector, "")
        // .def("", &VRCompCuda::VietorisRipsArgsGenerator::, "")
        ; 

    m.def("VertFiltCompCuda__vert_filt_comp_calculate_persistence_args", &VertFiltCompCuda::vert_filt_comp_calculate_persistence_args, "compute args for calculate_persistence from simplicial complex definition(CUDA)");
    m.def("VertFiltCompCuda__vert_filt_persistence_single", &VertFiltCompCuda::vert_filt_persistence_single, "");
    m.def("VertFiltCompCuda__vert_filt_persistence_batch", &VertFiltCompCuda::vert_filt_persistence_batch, "");

}

#endif