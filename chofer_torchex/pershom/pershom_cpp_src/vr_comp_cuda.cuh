#pragma once

#include <ATen/ATen.h>


using namespace at;


namespace VRCompCuda {
    std::vector<std::vector<Tensor>> vr_persistence(
        const Tensor& point_cloud, 
        int64_t max_dimension, 
        double max_ball_radius,          
        const std::string & metric);

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

    class PointCloud2VR
    {
        public:

        at::TensorOptions tensopt_real;
        at::TensorOptions tensopt_int; 
        // Type * RealType; TODO: delete 
        // Type * IntegerType; TODO: delete 
        // ScalarType IntegerScalarType = ScalarType::Long; TODO: delete 

        Tensor point_cloud; 
        int64_t max_dimension;
        double max_ball_radius; 

        std::function<Tensor(const Tensor &)> get_distance_matrix; 
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
        
        PointCloud2VR(const std::function<Tensor(const Tensor &)> get_distance_matrix)
            : get_distance_matrix(get_distance_matrix){}

        std::vector<Tensor> operator()(
            const Tensor & point_cloud, 
            int64_t max_dimension, 
            double max_ball_radius);

        void init_state(
            const Tensor & point_cloud, 
            int64_t max_dimension, 
            double max_ball_radius
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

    PointCloud2VR PointCloud2VR_factory(const std::string & distance);
}
