#include <torch/extension.h>
// #include <ATen/cuda/CUDAApplyUtils.cuh>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <thrust/sort.h>
// #include <thrust/unique.h>
// #include <thrust/device_ptr.h>

#include <vector>
#include <limits>
#include <future>

#include "vertex_filtration_comp_cuda.h"
#include "vr_comp_cuda.cuh"
#include "calc_pers_cuda.cuh"


using namespace torch;


namespace VertFiltCompCuda {
    std::vector<Tensor> vert_filt_comp_calculate_persistence_args(
        const Tensor & vertex_filtration, 
        const std::vector<Tensor> & boundary_info)
    {
        // initialize helper variables...
        auto ret = std::vector<Tensor>();

        auto tensopt_real = torch::TensorOptions()
            .dtype(vertex_filtration.dtype())
            .device(vertex_filtration.device());  

        auto tensopt_int = torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(vertex_filtration.device());

        Tensor num_simplices_by_dim; 
        {
            std::vector<long> tmp = {vertex_filtration.size(0)};

            for (auto const& t: boundary_info){
                tmp.push_back(t.size(0));
            }

            num_simplices_by_dim = torch::tensor(tmp);
        }
        auto num_simplices_up_to_dim = num_simplices_by_dim.cumsum(0); 
        auto max_dim = num_simplices_by_dim.size(0)-1; 

        // generate simplex dimensionality vector...
        auto simplex_dim = torch::zeros({num_simplices_up_to_dim[-1].item<long>()}, tensopt_int); 
        for (int i = 0; i < max_dim; i++){
            simplex_dim.slice(
                0,
                num_simplices_up_to_dim[i].item<long>(), 
                num_simplices_up_to_dim[i+1].item<long>()
            ).fill_(i+1);
            
        }

        // generate filtration values vector
        Tensor filt_val_vec;
        {
            auto tmp = std::vector<Tensor>();
            tmp.push_back(vertex_filtration);
            for (auto const& bi: boundary_info){
                auto v = tmp.back(); 
                v = v.unsqueeze(0).expand({bi.size(0), -1});
                v = v.gather(1, bi);

                v = std::get<0>(v.max(1));

                tmp.push_back(v);
            }

            filt_val_vec = torch::cat(tmp, 0); 
        }

        // adapt simplex ids from dimension-wise ids to global ids
        std::vector<Tensor> boundary_info_global_id;
        {
            auto tmp = std::vector<Tensor>();
            tmp.push_back(boundary_info.at(0)); 

            for (int i = 1; i < boundary_info.size(); i++){
                tmp.push_back(
                    boundary_info.at(i) + num_simplices_up_to_dim[i-1]
                );
            }

            boundary_info_global_id = tmp; 
        }

        // do sorting with epsilon hack ...
        Tensor sorted_filt_val_vec, perm_inv, perm; 
        {
            auto hack_add = simplex_dim.to(filt_val_vec.dtype()) * 100 * std::numeric_limits<float>::epsilon();
            auto tmp = filt_val_vec + hack_add; 
            auto sort_res = tmp.sort(0); 
            sorted_filt_val_vec = std::get<0>(sort_res);
            perm = std::get<1>(sort_res); 

            sort_res = perm.sort();
            perm_inv = std::get<1>(sort_res); 

            sorted_filt_val_vec = sorted_filt_val_vec - hack_add.index_select(0, perm); 
        }

        // transfer simplex ids to filtration-based ids ...
        {
            auto tmp = std::vector<Tensor>();
            for (auto const& bi: boundary_info_global_id){
                auto perm_inv_expanded = perm_inv.unsqueeze(0).expand({bi.size(0), -1});
                auto bi_new = perm_inv_expanded.gather(1, bi); 
                bi_new = std::get<0>(bi_new.sort(1, true)); 
                tmp.push_back(bi_new); 
            }

            boundary_info_global_id = tmp; 
        }

        // create boundary array ... 
        auto ba = torch::empty(
            {num_simplices_up_to_dim[-1].item<long>(), 
            (max_dim+1)*2}, 
            tensopt_int
        );
        ba.fill_(-1); 

        for (int i = 0; i < boundary_info_global_id.size(); i++){
            ba.slice(
                0, 
                num_simplices_up_to_dim[i].item<long>(),
                num_simplices_up_to_dim[i+1].item<long>()
            ).slice(
                1, 0, i+2
            ) = boundary_info_global_id.at(i);
        }
        // final sorting ... 

        ba = ba.index_select(0, perm);
        simplex_dim = simplex_dim.index_select(0, perm);

        // compressing 
        auto i = simplex_dim.gt(0).nonzero().squeeze(); 
        auto ba_row_i_to_bm_col_i = torch::arange(0, ba.size(0), tensopt_int);

        ba_row_i_to_bm_col_i = ba_row_i_to_bm_col_i.index_select(0, i); 
        ba = ba.index_select(0, i); 

        ret.push_back(ba); 
        ret.push_back(ba_row_i_to_bm_col_i); 
        ret.push_back(simplex_dim); 
        ret.push_back(sorted_filt_val_vec); 

        return ret;
    }


    // //TODO compare to the 'calculate_persistence_output_to_barcode_tensors'
    // // function in vr_comp_cuda.cu and refactor
    // Tensor read_non_essential_barcode(
    //     const Tensor & barcode, 
    //     const Tensor & sorted_filtration_values)
    // {
    //     Tensor ret; 
    //     if (barcode.size(0) == 0)
    //     {
    //        ret = torch::empty({0, 2}, sorted_filtration_values.options()); 
    //     }
    //     else
    //     {
    //         auto v = sorted_filtration_values
    //             .unsqueeze(0)
    //             .expand({barcode.size(0), -1});
    //         ret = v.gather(1, barcode); 

    //         auto i = ret.slice(1, 0, 1).ne(ret.slice(1, 1, 2)); 
    //         i = i.nonzero().squeeze();

    //         if (i.size(0) == 0){
    //             ret =  torch::empty({0, 2}, sorted_filtration_values.options()); 
    //         }
    //         else
    //         {
    //             ret = ret.index_select(0, i);
    //         } 
    //     }

    //     return ret;
    // }


    // Tensor read_essential_barcode(
    //     const Tensor & barcode, 
    //     const Tensor & sorted_filtration_values)
    // {
    //     Tensor ret; 
    //     if (barcode.size(0) == 0)
    //     {
    //        ret = torch::empty({0, 1}, sorted_filtration_values.options()); 
    //     }
    //     else
    //     {
    //         auto v = sorted_filtration_values
    //             .unsqueeze(0)
    //             .expand({barcode.size(0), -1});
    //         ret = v.gather(1, barcode); 
    //     }

    //     return ret;      
    // }


    // std::vector<std::vector<Tensor>> read_barcode_from_birth_death_times(
    //     const std::vector<std::vector<Tensor>>& calculate_persistence_output,
    //     const Tensor & sorted_filtration_values)
    // {   
    //     auto ret_non_ess = std::vector<Tensor>();
    //     auto ret_ess     = std::vector<Tensor>();
    //     auto ret         = std::vector<std::vector<Tensor>>();


    //     auto non_ess_barcodes = calculate_persistence_output.at(0);
    //     auto ess_barcodes     = calculate_persistence_output.at(1); 

    //     for (auto const& b: non_ess_barcodes){
    //         ret_non_ess.push_back(
    //             read_non_essential_barcode(
    //                 b, 
    //                 sorted_filtration_values
    //             )
    //         );
    //     }

    //     for (auto const& b: ess_barcodes){
    //         ret_ess.push_back(
    //             read_essential_barcode(
    //                 b, 
    //                 sorted_filtration_values
    //             )
    //         );
    //     }

    //     ret.push_back(ret_non_ess);
    //     ret.push_back(ret_ess); 

    //     return ret; 
    // }


    std::vector<std::vector<Tensor>> vert_filt_persistence_single(
        const Tensor & vertex_filtration, 
        const std::vector<Tensor> & boundary_info)
    {
        auto r = vert_filt_comp_calculate_persistence_args(
            vertex_filtration, 
            boundary_info
        );

        auto ba = r.at(0);
        auto ba_row_i_to_bm_col_i = r.at(1);
        auto simplex_dim = r.at(2);
        auto sorted_filtration_values = r.at(3); 

        auto calc_pers_output = CalcPersCuda::calculate_persistence(
            ba, 
            ba_row_i_to_bm_col_i, 
            simplex_dim, 
            boundary_info.size(), 
            -1 
        );
        // VRCompCuda::calculate_persistence_output_to_barcode_tensors
        // read_barcode_from_birth_death_times
        return VRCompCuda::calculate_persistence_output_to_barcode_tensors(
            calc_pers_output, 
            sorted_filtration_values
        );
    }


    std::vector<std::vector<std::vector<Tensor>>> vert_filt_persistence_batch(
        const std::vector<std::tuple<Tensor, std::vector<Tensor>>> & batch
    )
    {
        auto futures = std::vector<std::future<std::vector<std::vector<Tensor>>>>();
        for (auto & arg: batch){

            futures.push_back(
                std::async(
                    std::launch::async, 
                    [=]{
                        return vert_filt_persistence_single(
                            std::get<0>(arg), 
                            std::get<1>(arg)
                        );
                    }
                )
            );
        }

        auto ret = std::vector<std::vector<std::vector<Tensor>>>();
        for (auto & fut: futures){
            ret.push_back(
                fut.get()
            );
        }

        return ret;
    }
}
