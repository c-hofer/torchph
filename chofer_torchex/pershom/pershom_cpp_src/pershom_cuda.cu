#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


using namespace at;


#pragma region find_merge_pairings


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


class NoPairsException{
public:
  NoPairsException() {}
 ~NoPairsException() {}
};


//FIXME make this method template to get ride of hardcoded int32_t 
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
  output = output.view(IntList({output.size(0)/2, 2}));

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

    Tensor merge_pairs;
    if (pairing_tensors.size() != 0){      
      merge_pairs = cat(pairing_tensors, 0);
      merge_pairs = merge_pairs.slice(0, 0,  max_pairs);
    }
    else{
      throw NoPairsException();
    }

    // std::cout << merge_pairs << std::endl;
    // std::cout <<"End"<<std::endl;

   return merge_pairs;
}


#pragma endregion 


#pragma region merge_columns


namespace{


  template <typename scalar_t>
  __device__ void merge_one_column_s(
    scalar_t* p_merger, 
    scalar_t* p_target, // the position of the target column, set to -1
    scalar_t* p_target_cache, // contains the copied values of target column 
    int boundary_array_size_1, 
    int* d_boundary_array_needs_resize 
  ){    
    // Assertion: descending_sorted_boundary_array[:, -1] == -1 

    int p_target_increment_count = 0;

    while (true){
      if (*p_merger == -1 && *p_target_cache == -1){
        // both are -1, we have reached the end of meaningful entries -> break
        break;
      }

      if (*p_merger == *p_target_cache){
        // both values are the same but not -1 -> we eliminate 
        p_target_cache++;
        p_merger++;
      }
      else {

        if (*p_merger > *p_target_cache){
          //merger value is greater -> we take it 
          *p_target = *p_merger;
          p_merger++;
        }
        else
        {
          //target value is greate -> we take it 
          *p_target = *p_target_cache;
          p_target_cache++;
        }

        p_target++;  
        p_target_increment_count += 1;
      }          
    }

    if (p_target_increment_count > boundary_array_size_1/2){
      *d_boundary_array_needs_resize = 1; 
    }
  }


  template <typename scalar_t>
  __global__ void merge_columns_cuda_kernel(
      scalar_t* descending_sorted_boundary_array,
      size_t descending_sorted_boundary_array_size_1, 
      scalar_t* cache, 
      int64_t* merge_pairings,
      size_t merge_pairings_size_0, 
      int* d_boundary_array_needs_resize
  ){
    const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;   

    if (thread_id < merge_pairings_size_0){  

      const int filt_id_merger = merge_pairings[thread_id * 2];
      const int filt_id_target = merge_pairings[thread_id * 2 + 1];

      merge_one_column_s<int32_t>(
        descending_sorted_boundary_array + (filt_id_merger * descending_sorted_boundary_array_size_1),
        descending_sorted_boundary_array + (filt_id_target * descending_sorted_boundary_array_size_1),
        cache + thread_id, 
        descending_sorted_boundary_array_size_1,
        d_boundary_array_needs_resize
      );
    }
  }
  

} //namespace


Tensor resize_boundary_array(
  Tensor descending_sorted_boundary_array){
    auto tmp = empty_like(descending_sorted_boundary_array);
    tmp.fill_(-1);
    auto new_ba = cat(TensorList({descending_sorted_boundary_array, tmp}), 1);
    return new_ba.contiguous();
}


Tensor merge_columns_cuda(
  Tensor descending_sorted_boundary_array, 
  Tensor merge_pairings){
    const int threads_per_block = 256;
    const int blocks = merge_pairings.size(0)/threads_per_block + 1;

    auto targets = merge_pairings.slice(1, 1).squeeze();
    // std::cout << "merge_columns_cuda --> targets:" << std::endl << targets << std::endl;

    // fill cache for merging 
    //TODO optimize: we do not need all columns it is enough to take des...array.size(1)/2 + 1
    auto cache = descending_sorted_boundary_array.index_select(0, targets);

    // reset content of target columns 
    descending_sorted_boundary_array.index_fill_(0, targets, -1);

    // std::cout << "merge_columns_cuda --> cache:" << std::endl << cache << std::endl;
    
    // bool boundary_array_needs_resize = false; //Is this save?
    auto size = sizeof(int);
    int boundary_array_needs_resize = 0;
    int* h_boundary_array_needs_resize = &boundary_array_needs_resize;
    int* d_boundary_array_needs_resize;
    cudaMalloc(&d_boundary_array_needs_resize, size);
    cudaMemcpy(d_boundary_array_needs_resize, h_boundary_array_needs_resize, size, cudaMemcpyHostToDevice);

    merge_columns_cuda_kernel<int32_t><<<blocks, threads_per_block>>>(
      descending_sorted_boundary_array.data<int32_t>(), 
      descending_sorted_boundary_array.size(1), 
      cache.data<int32_t>(),
      merge_pairings.data<int64_t>(), 
      merge_pairings.size(0), 
      d_boundary_array_needs_resize
    );
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_boundary_array_needs_resize, d_boundary_array_needs_resize, size, cudaMemcpyDeviceToHost);
    // std::cout << "merge_columns_cuda --> boundary_array_needs_resize:" << (*h_boundary_array_needs_resize ? "True" : "False") << std::endl;


    if (*h_boundary_array_needs_resize == 1){
      // std::cout << "Resizing ..." << std::endl;
      descending_sorted_boundary_array = resize_boundary_array(descending_sorted_boundary_array);
    }

    cudaFree(d_boundary_array_needs_resize);
    
    return descending_sorted_boundary_array;
  }


#pragma endregion


#pragma region read_points
Tensor read_points_cuda(
  Tensor reduced_descending_sorted_boundary_array);


#pragma endregion 


Tensor calculate_persistence_cuda(  
  Tensor descending_sorted_boundary_array, 
  Tensor column_dimension,
  int max_pairs) {

  // std::cout << "calculate_persistence_cuda --> descending_sorted_boundary_array " << std::endl << descending_sorted_boundary_array << std::endl;

  Tensor pivots, merge_pairings;
  while(true){
    pivots = descending_sorted_boundary_array.slice(1, 0, 1);

    try{
      merge_pairings = find_merge_pairings_cuda(pivots, max_pairs);
      // std::cout << merge_pairings.size(0) << std::endl;
    }
    catch(NoPairsException& e){
      std::cout << "Reached end of reduction" << std::endl;
      break;
    }
    

    // std::cout << "calculate_persistence_cuda --> merge_pairings:" << std::endl << merge_pairings << std::endl;
    
    descending_sorted_boundary_array = merge_columns_cuda(descending_sorted_boundary_array, merge_pairings);

    // std::cout << "calculate_persistence_cuda --> descending_sorted_boundary_array " << std::endl << descending_sorted_boundary_array << std::endl;
  }

  return descending_sorted_boundary_array;

}
