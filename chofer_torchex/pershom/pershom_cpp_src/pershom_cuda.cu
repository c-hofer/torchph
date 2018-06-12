#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


using namespace at;


namespace {

template <typename scalar_t>
__global__ void find_left_plateau_indices_cuda_kernel(
  scalar_t* __restrict__ input,
  scalar_t* __restrict__ output, 
  size_t input_size){ 

    const int index_middle = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_left   = index_middle - 1;
    const int index_right  = index_middle + 1;
    
    if (index_right < input_size){

      const int value_left = (index_left != -1) ? input[index_left] : -1; //OPTIMIZE: if we could
      // pad input with -1 on the left this conditional would be obsolete
      const int value_middle = input[index_middle];
      const int value_right = input[index_right];
      if (value_left != value_middle
          && 
          value_middle == value_right){
        output[index_middle] = index_middle;
      }
    }    
  }


template <typename scalar_t>
__global__ void find_right_plateau_indices_cuda_kernel(
  scalar_t* __restrict__ input,
  scalar_t* __restrict__ output, 
  size_t input_size){ 

    const int index_left   = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_middle = index_left + 1;
    const int index_right  = index_middle + 1;
    
    if (index_middle < input_size){

      const int value_left = input[index_left];
      const int value_middle = input[index_middle];
      const int value_right = (index_right < input_size) ? input[index_right] : (value_middle+1);//OPTIMIZE: if we could
      // pad input with inf on the right this conditional would be obsolete
      if (value_left == value_middle
          && 
          value_middle != value_right){
        output[index_middle] = index_middle;
      }
    }    
  }


} // namespace

Tensor find_slicing_indices(
    Tensor input) {
  Tensor output = zeros_like(input).fill_(-1);
  
  const int threads_per_block = 256;
  const int blocks = input.size(0)/threads_per_block + 1;

  find_left_plateau_indices_cuda_kernel<int32_t><<<blocks, threads_per_block>>>(
    input.data<int32_t>(), 
    output.data<int32_t>(),
    input.size(0));

  find_right_plateau_indices_cuda_kernel<int32_t><<<blocks, threads_per_block>>>(
    input.data<int32_t>(), 
    output.data<int32_t>(),
    input.size(0));

  output = output.masked_select(output.ge(0));
  output.resize_(IntList({output.size(0)/2, 2}));

  return output;
}


Tensor find_merge_pairings_cuda(
  Tensor pivots,
  int max_pairs){

    // std::cout << pivots << std::endl;

    auto sort_res = pivots.sort(0);
    auto sort_val = std::get<0>(sort_res);
    auto sort_ind = std::get<1>(sort_res);

    // remove columns with undefined pivot (i.e. -1)
    auto mask = sort_val.ge(0);
    sort_val = sort_val.masked_select(mask);
    sort_ind = sort_ind.masked_select(mask);

    // std::vector<Tensor> l({sort_val, sort_ind.type_as(sort_val)});
    // std::cout << stack(l, 1) << std::endl;

    auto slicings = find_slicing_indices(sort_val);
    // std::cout << slicings << std::endl;

    int pairing_counter = 0;
    std::vector<Tensor> pairing_tensors; 
    for (int i=0; i<slicings.size(0); i++){

      if (pairing_counter > max_pairs){
        break;
      }

      auto slicing_i = slicings[i];
      auto begin = Scalar(slicing_i[0]).to<int>(); //OPTIMIZE: can this conversion be improved?
      auto end = Scalar(slicing_i[1]).to<int>() + 1;
      auto slice = sort_ind.slice(0, begin, end);
      slice = std::get<0>(slice.sort(0));

      auto col_2 = slice.slice(0, 1);
      auto col_1 = slice[0].expand_as(col_2);
      auto pairing_tensor = stack(std::vector<Tensor>({col_1, col_2}), 1);

      pairing_counter += pairing_tensor.size(0);

      pairing_tensors.push_back(pairing_tensor);
    }

    auto merge_pairs = cat(pairing_tensors, 0);
    merge_pairs = merge_pairs.slice(0, 0,  max_pairs);

    // std::cout << merge_pairs << std::endl;
    // std::cout <<"End"<<std::endl;


   return merge_pairs;
}


void merge_columns_cuda(
  Tensor descending_sorted_boundary_array, 
  Tensor merge_pairs);


Tensor read_points_cuda(
  Tensor reduced_descending_sorted_boundary_array);


Tensor calculate_persistence_cuda(  
  Tensor descending_sorted_boundary_array, 
  Tensor column_dimension,
  int max_pairs) {
  
  auto merge_pairings = find_merge_pairings_cuda(descending_sorted_boundary_array, max_pairs);
  std::cout << merge_pairings << std::endl;

  return descending_sorted_boundary_array;
}
