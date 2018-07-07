#pragma once

#include <ATen/ATen.h>


using namespace at;


namespace VRCompCuda {
    std::vector<std::vector<Tensor>> vr_l1_persistence_checked(const Tensor& point_cloud, double max_ball_radius);

    void write_combinations_table_to_tensor(
        const Tensor& out, 
        const int64_t out_row_offset, 
        const int64_t additive_constant, 
        const int64_t max_n, 
        const int64_t r
        );
}