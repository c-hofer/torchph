#pragma once


#include <torch/extension.h>


using namespace torch;


namespace VRCompCuda {
    std::vector<std::vector<Tensor>> vr_persistence(
        const Tensor& distance_matrix, 
        int64_t max_dimension, 
        double max_ball_diameter);

    std::vector<std::vector<Tensor>> vr_persistence_l1(
        const Tensor& point_cloud, 
        int64_t max_dimension, 
        double max_ball_diameter);

    void write_combinations_table_to_tensor(
        const Tensor& out, 
        const int64_t out_row_offset, 
        const int64_t additive_constant, 
        const int64_t max_n, 
        const int64_t r);

    Tensor co_faces_from_combinations(
        const Tensor & combinations, 
        const Tensor & faces);

    Tensor l1_norm_distance_matrix(const Tensor & points); 
    
    Tensor l2_norm_distance_matrix(const Tensor & points);

    class VietorisRipsArgsGenerator
    {
        public:

        at::TensorOptions tensopt_real;
        at::TensorOptions tensopt_int; 

        Tensor distance_matrix; 
        int64_t max_dimension;
        double max_ball_diameter; 

        std::vector<Tensor> boundary_info_non_vertices;
        std::vector<Tensor> filtration_values_by_dim; 
        std::vector<int64_t> n_simplices_by_dim; 

        Tensor simplex_dimension_vector; 
        Tensor filtration_values_vector_without_vertices; 
        Tensor filtration_add_eps_hack_values;

        Tensor sort_indices_without_vertices;
        Tensor sort_indices_without_vertices_inverse; 
        Tensor sorted_filtration_values_vector;

        Tensor boundary_array; 

        Tensor ba_row_i_to_bm_col_i_vector; 

        std::vector<Tensor> operator()(
            const Tensor & distance_matrix, 
            int64_t max_dimension, 
            double max_ball_diameter);

        void init_state(
            const Tensor & distance_matrix, 
            int64_t max_dimension, 
            double max_ball_diameter
            );

        void make_boundary_info_edges(); 
        void make_boundary_info_non_edges();
        void make_simplex_ids_compatible_within_dimensions();
        void make_simplex_dimension_vector(); 
        void make_filtration_values_vector_without_vertices();
        void do_filtration_add_eps_hack();
        void make_sorting_infrastructure(); 
        void undo_filtration_add_eps_hack(); 
        void make_sorted_filtration_values_vector(); 
        void make_boundary_array_rows_unsorted(); 
        void apply_sorting_to_rows();
        void make_ba_row_i_to_bm_col_i_vector(); 
    };


    std::vector<std::vector<Tensor>> calculate_persistence_output_to_barcode_tensors(
        const std::vector<std::vector<Tensor>>& calculate_persistence_output,
        const Tensor & filtration_values);
}

