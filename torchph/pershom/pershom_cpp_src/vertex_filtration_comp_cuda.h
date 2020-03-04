#pragma once


#include <torch/extension.h>


using namespace torch;


namespace VertFiltCompCuda
{
    std::vector<Tensor> vert_filt_comp_calculate_persistence_args(
        const Tensor & vertex_filtration, 
        const std::vector<Tensor> & boundary_info);

    std::vector<std::vector<Tensor>> vert_filt_persistence_single(
        const Tensor & vertex_filtration, 
        const std::vector<Tensor> & boundary_info);

    std::vector<std::vector<std::vector<Tensor>>> vert_filt_persistence_batch(
        const std::vector<std::tuple<Tensor, std::vector<Tensor>>> & batch
    );
}