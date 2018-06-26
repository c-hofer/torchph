#include <stdio.h>
#include <iostream>
#include <fstream> 
// #include <vector>
#include <string>
#include <tuple>
#include <torch/torch.h>
#include <dlfcn.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <google/profiler.h>

using namespace at;


std::vector<std::vector<Tensor>> calculate_persistence(
    Tensor descending_sorted_boundary_array,
    Tensor ind_not_reduced, 
    Tensor column_dimension,
    int max_pairs,
    int max_dimension=-1);

Tensor find_merge_pairings(Tensor pivots, int max_pairs);

std::tuple<Tensor, Tensor, Tensor, int> read_profiling_data(){

    std::tuple<Tensor, Tensor, int> ret;
    std::ifstream inputFile("data/boundary_array.txt");

    std::vector<int32_t> boundary_array_data;

    if (inputFile.good()) {
        
        int32_t current_number = 0;
        while (inputFile >> current_number){
            boundary_array_data.push_back(current_number);
        }

        inputFile.close();
    }


    inputFile = std::ifstream("data/boundary_array_size.txt");
    std::vector<int> boundary_array_size;

    if (inputFile.good()) {
        
        int current_number = 0;
        while (inputFile >> current_number){
            boundary_array_size.push_back(current_number);
        }

        inputFile.close();
    }

    int32_t *d_boundary_array_data;

    // Sends data to device
    auto size = boundary_array_size[0]*boundary_array_size[1]*sizeof(int32_t);
    cudaMalloc((void**) &d_boundary_array_data, size);
    cudaMemcpy(d_boundary_array_data, &boundary_array_data[0], size, cudaMemcpyHostToDevice);   

    auto boundary_array = CUDA(kInt).tensorFromBlob(d_boundary_array_data, {boundary_array_size[0], boundary_array_size[1]});
    boundary_array = boundary_array.clone();
    cudaFree(d_boundary_array_data); 
    // auto boundary_array = CPU(kInt).tensorFromBlob(&boundary_array_data[0], {boundary_array_size[0], boundary_array_size[1]});


    inputFile = std::ifstream("data/ind_not_reduced.txt");
    std::vector<int64_t> ind_not_reduced_data;

    if (inputFile.good()) {
        
        int64_t current_number = 0;
        while (inputFile >> current_number){
            ind_not_reduced_data.push_back(current_number);
        }

        inputFile.close();
    }

    int64_t *d_ind_not_reduced_data;
    size = ind_not_reduced_data.size()*sizeof(int64_t);
    cudaMalloc((void**) &d_ind_not_reduced_data, size);
    cudaMemcpy(d_ind_not_reduced_data, &ind_not_reduced_data[0], size, cudaMemcpyHostToDevice);   

    auto ind_not_reduced = CUDA(kLong).tensorFromBlob(d_ind_not_reduced_data, {ind_not_reduced_data.size()});
    ind_not_reduced = ind_not_reduced.clone();
    cudaFree(d_ind_not_reduced_data);


    inputFile = std::ifstream("data/column_dimension.txt");
    std::vector<int> column_dimension_data;

    if (inputFile.good()) {
        
        int current_number = 0;
        while (inputFile >> current_number){
            column_dimension_data.push_back(current_number);
        }

        inputFile.close();
    }

    int32_t *d_column_dimension_data;
    size = column_dimension_data.size()*sizeof(int32_t);
    cudaMalloc((void**) &d_column_dimension_data, size);
    cudaMemcpy(d_column_dimension_data, &column_dimension_data[0], size, cudaMemcpyHostToDevice);   

    auto column_dimension = CUDA(kInt).tensorFromBlob(d_column_dimension_data, {column_dimension_data.size()});
    column_dimension = column_dimension.clone();
    cudaFree(d_column_dimension_data);
    // auto column_dimension = CPU(kInt).tensorFromBlob(&column_dimension_data[0], {column_dimension_data.size()});


    inputFile = std::ifstream("data/max_dimension.txt");
    int max_dimension;

    if (inputFile.good()) {
        
        inputFile >> max_dimension;

        inputFile.close();
    }

    return std::make_tuple(boundary_array, ind_not_reduced, column_dimension, max_dimension);
    
}

void sorting(Tensor pivots){
    pivots.sort(0);
}

int main()
{
    dlopen("libcaffe2_gpu.so", RTLD_NOW);

    auto bm_ind_dim_maxdim = read_profiling_data();

    auto bm = std::get<0>(bm_ind_dim_maxdim);
    auto ind_not_reduced = std::get<1>(bm_ind_dim_maxdim);
    auto col_dim = std::get<2>(bm_ind_dim_maxdim);
    auto max_dim = std::get<3>(bm_ind_dim_maxdim);

    // auto pivots = bm.slice(1, 0, 1).contiguous();

    ProfilerStart("profile.google-pprof");
    for (int i=0; i<10; i++){
    // auto pairs = find_merge_pairings(pivots, 100000);
    auto ret = calculate_persistence(bm.clone(), ind_not_reduced.clone(), col_dim.clone(), max_dim);
    }
    ProfilerStop();

    // pivots = pivots.toBackend(Backend::CPU);
    // std::cout  << Scalar(pivots[0][0]).to<int>() << std::endl;

    return 0;
}