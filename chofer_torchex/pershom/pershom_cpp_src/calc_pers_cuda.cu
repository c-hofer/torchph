#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

#include "tensor_utils.cuh"
#include "param_checks_cuda.cuh"


using namespace at;

//TODO do proper namespacing and make header file

//TODO resolve template chaos should we always use long and never int? YES

//TODO change from by value args to by reference args 

//TODO remove corresponding cpp file and insert checked call wrappers in cu file

//TODO refactor python bindings in new file 

//TODO more assertions 
#pragma region find_merge_pairings


namespace CalcPersCuda{


namespace {

template <typename scalar_t>
__global__ void find_left_slicings_indices_cuda_kernel(
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
__global__ void find_right_slicings_indices_cuda_kernel(
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
        output[index_middle] = index_middle + 1;
      }
    }    
  }

} // namespace


/**
 * @brief Finds the indices for slicing the sorted pivots values. 
 * Example:
 *    pivots.sort(0)[0] = [-1, -1, 2, 2, 2, 4, 4] -> [[2, 4], [5, 6]]
 * 
 * @tparam scalar_t 
 * @param pivots 
 * @return Tensor return.dtype() == scalar_t
 */
template <typename scalar_t>
Tensor find_slicing_indices_cuda_kernel_call(
    Tensor pivots) {
  Tensor output = zeros_like(pivots).fill_(-1);
  const int threads_per_block = 256;
  const int blocks = pivots.size(0)/threads_per_block + 1;

  find_left_slicings_indices_cuda_kernel<scalar_t><<<blocks, threads_per_block>>>(
    pivots.data<scalar_t>(), 
    output.data<scalar_t>(),
    pivots.size(0));

  find_right_slicings_indices_cuda_kernel<scalar_t><<<blocks, threads_per_block>>>(
    pivots.data<scalar_t>(), 
    output.data<scalar_t>(),
    pivots.size(0));

  output = output.masked_select(output.ge(0));
  output = output.view(IntList({output.size(0)/2, 2}));

  return output;
}  


namespace {

/**
 * @brief Implements a batch version of traditional slice for 
 * a input vector and a tensor which defines the slicings.
 * The output is a then of dimension
 * slicings.size(0) x (slicings[:, 1] - slicings[:, 0]).max()
 * 
 * @param p_input 
 * @param p_slicings 
 * @param p_return_value 
 * @param return_value_size_0 
 * @param return_value_size_1 
 */
__global__ void extract_slicings_cuda_kernel(
  int64_t* p_input,
  int32_t* p_slicings, 
  int64_t* p_return_value, 
  int64_t return_value_size_0, 
  int64_t return_value_size_1){
    
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < return_value_size_0){
      auto p_return_value_row = p_return_value + thread_id * return_value_size_1;
      const int slice_start = *(p_slicings + (thread_id * 2));
      const int slice_end = *(p_slicings + (thread_id * 2) + 1);

      for (int i = 0; i < slice_end - slice_start; i++){
        *(p_return_value_row + i) = *(p_input + slice_start + i);
      }
    }
}

/**
 * @brief Intended to be used on the output of 
 * extract_slicings_cuda_kernel. It reformats 
 * extraextracted_slices row-wise to merge-pairs 
 * format. E.g. 
 * row_i = [1, 2, 3] -> [[1,2], [1,3]]
 * 
 * @param extracted_slices 
 * @param extracted_slices_size_0 
 * @param extracted_slices_size_1 
 * @param lengths 
 * @param row_offset_for_thread 
 * @param return_value 
 */
__global__ void format_extracted_sorted_slicings_to_merge_pairs_kernel(
  int64_t* extracted_slices, 
  int64_t extracted_slices_size_0,
  int64_t extracted_slices_size_1,
  int32_t* lengths, 
  int64_t* row_offset_for_thread,
  int64_t* return_value){
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < extracted_slices_size_0){
      const int length = *(lengths + thread_id);
      const int row_offset = (thread_id > 0) ? *(row_offset_for_thread + thread_id - 1) : 0;
      auto extracted_slices_row = extracted_slices + thread_id*extracted_slices_size_1; 
      auto const first_col_value = *extracted_slices_row;

      auto return_value_row = return_value + 2*row_offset;
      for (int i = 0; i < length - 1; i++){
        *(return_value_row) = first_col_value;
        *(return_value_row + 1) = *(extracted_slices_row + i + 1);
        return_value_row = return_value_row + 2;
      }
    }
  }

} //namespace


Tensor sorted_pivot_indices_to_merge_pairs_cuda_kernel_call(Tensor input, Tensor slicings){
  // ASSERTION input.dtype() == int64
  // ASSERTION slicings.dtype() == int32
  // ASSERTION all(input.ge(0))
  // ASSERTION all(slicings.ge(0))
  // ASSERTION all(slicings[:, 0].leq(slicings[:, 1]))
  // ASSERTION slicings[:, 1].max() < input.size(0)

  auto lengths = (slicings.slice(1, 1, 2) - slicings.slice(1, 0, 1)).contiguous();
  auto max_lengths = Scalar(lengths.max()).to<int>(); 
  Tensor extracted_slicings = input.type().tensor({slicings.size(0), max_lengths});
  extracted_slicings.fill_(std::numeric_limits<int64_t>::max());

  const int threads_per_block_apply_slicings = 256;
  const int blocks_apply_slicings = slicings.size(0)/threads_per_block_apply_slicings + 1;
  extract_slicings_cuda_kernel<<<threads_per_block_apply_slicings, blocks_apply_slicings>>>(
    input.data<int64_t>(), 
    slicings.data<int32_t>(),
    extracted_slicings.data<int64_t>(),
    extracted_slicings.size(0),
    extracted_slicings.size(1)
  );

  auto extracted_slicings_sorted = std::get<0>(extracted_slicings.sort(1)).contiguous();

  auto lengths_minus_1 = lengths - lengths.type().scalarTensor(1);  
  auto row_offset_for_thread = lengths_minus_1.cumsum(0);

  auto merge_pairings_size_0 = Scalar(row_offset_for_thread[-1][0]).to<int>();
  auto merge_pairings = input.type().tensor({merge_pairings_size_0, 2});
  merge_pairings.fill_(-1);

  const int threads_per_block = 256;
  const int blocks = extracted_slicings_sorted.size(0)/threads_per_block + 1;

  format_extracted_sorted_slicings_to_merge_pairs_kernel<<<threads_per_block, blocks>>>(
      extracted_slicings_sorted.data<int64_t>(), 
      extracted_slicings_sorted.size(0),
      extracted_slicings_sorted.size(1), 
      lengths.data<int32_t>(), 
      row_offset_for_thread.data<int64_t>(),
      merge_pairings.data<int64_t>()
  );

  return merge_pairings;
  
}

class NoPairsException{
  public:
    NoPairsException() {}
    ~NoPairsException() {}
};


Tensor find_merge_pairings(
  Tensor pivots,
  int max_pairs = -1 ){

    CHECK_INPUT(pivots);
    assert(pivots.type().scalarType() == ScalarType::Int);

    if (max_pairs < 1){
      max_pairs = std::numeric_limits<int>::max();
    }
    auto sort_res = pivots.sort(0);
    auto sort_val = std::get<0>(sort_res);
    auto sort_ind = std::get<1>(sort_res);

    auto slicings = find_slicing_indices_cuda_kernel_call<int32_t>(sort_val).contiguous();

    Tensor merge_pairs; 
    if (slicings.size(0) != 0){         

      merge_pairs = sorted_pivot_indices_to_merge_pairs_cuda_kernel_call(sort_ind, slicings);
      // We sort the pairs such that pairs with smaller index come first.
      // This improves performance???
      if (merge_pairs.size(0) > max_pairs){

        sort_res = merge_pairs.slice(1, 0, 1).sort(0);
        sort_ind = std::get<1>(sort_res);
        sort_ind = sort_ind.slice(0, 0, max_pairs).squeeze();

        merge_pairs = merge_pairs.index_select(0, sort_ind);
        merge_pairs = merge_pairs.contiguous();
      }
    }
    else{
      throw NoPairsException();
    }

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
    // Assertion: comp_desc_sort_ba[:, -1] == -1 

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
      scalar_t* comp_desc_sort_ba,
      size_t descending_sorted_boundary_array_size_1, 
      scalar_t* cache, 
      int64_t* merge_pairings,
      size_t merge_pairings_size_0, 
      int* d_boundary_array_needs_resize
  ){
    //ASSERTION: cache.size(1) == comp_desc_sort_ba.size(1)
    const int thread_id = blockIdx.x*blockDim.x + threadIdx.x;   

    if (thread_id < merge_pairings_size_0){  

      const int filt_id_merger = merge_pairings[thread_id * 2];
      const int filt_id_target = merge_pairings[thread_id * 2 + 1];

      merge_one_column_s<int32_t>(
        comp_desc_sort_ba + (filt_id_merger * descending_sorted_boundary_array_size_1),
        comp_desc_sort_ba + (filt_id_target * descending_sorted_boundary_array_size_1),
        cache + (thread_id * descending_sorted_boundary_array_size_1), 
        descending_sorted_boundary_array_size_1,
        d_boundary_array_needs_resize
      );
    }
  }
  

} //namespace


template <typename scalar_t>
void merge_columns_cuda_kernel_call(
  Tensor comp_desc_sort_ba,
  Tensor merge_pairings, 
  int* h_boundary_array_needs_resize
)
{
  const int threads_per_block = 32;
  const int blocks = merge_pairings.size(0)/threads_per_block + 1;

  auto targets = merge_pairings.slice(1, 1).squeeze();  
  
  // fill cache for merging ... 
  //  TODO optimize: we do not need all columns it is enough to take des...array.size(1)/2 + 1 
  //  ATTENTION if we do this we have to inform merge_columns_cuda_kernel about this!!!
  auto cache = comp_desc_sort_ba.index_select(0, targets);
  
  auto size = sizeof(int);
  int* d_boundary_array_needs_resize;
  cudaMalloc(&d_boundary_array_needs_resize, size);
  cudaMemcpy(d_boundary_array_needs_resize, h_boundary_array_needs_resize, size, cudaMemcpyHostToDevice);

  // reset content of target columns 
  comp_desc_sort_ba.index_fill_(0, targets, -1);

  merge_columns_cuda_kernel<int32_t><<<blocks, threads_per_block>>>(
    comp_desc_sort_ba.data<int32_t>(), 
    comp_desc_sort_ba.size(1), 
    cache.data<int32_t>(),
    merge_pairings.data<int64_t>(), 
    merge_pairings.size(0), 
    d_boundary_array_needs_resize
  );

  cudaDeviceSynchronize();
  cudaMemcpy(h_boundary_array_needs_resize, d_boundary_array_needs_resize, size, cudaMemcpyDeviceToHost);

  cudaFree(d_boundary_array_needs_resize);
}


Tensor resize_boundary_array(
  Tensor comp_desc_sort_ba){
    auto tmp = empty_like(comp_desc_sort_ba);
    tmp.fill_(-1);
    auto new_ba = cat(TensorList({comp_desc_sort_ba, tmp}), 1);
    return new_ba.contiguous();
}


Tensor merge_columns(
  Tensor comp_desc_sort_ba, 
  Tensor merge_pairings){   

    CHECK_INPUT(comp_desc_sort_ba);
    assert(comp_desc_sort_ba.type().scalarType() == ScalarType::Int);
    CHECK_INPUT(merge_pairings);
    assert(merge_pairings.type().scalarType() == ScalarType::Long);
    
    int boundary_array_needs_resize = 0;
    int* h_boundary_array_needs_resize = &boundary_array_needs_resize;    

    merge_columns_cuda_kernel_call<int32_t>(
      comp_desc_sort_ba,
      merge_pairings, 
      h_boundary_array_needs_resize
    );
  
    if (*h_boundary_array_needs_resize == 1){
      comp_desc_sort_ba = resize_boundary_array(comp_desc_sort_ba);
    }
    
    return comp_desc_sort_ba;
  }


#pragma endregion


#pragma region read_barcodes


std::vector<std::vector<Tensor> > read_barcodes(
  Tensor pivots, 
  Tensor simplex_dimension, 
  int max_dimension){

    CHECK_INPUT(pivots);
    assert(pivots.type().scalarType() == ScalarType::Int);
    CHECK_INPUT(simplex_dimension);
    assert(simplex_dimension.type().scalarType() == ScalarType::Int);
    std::vector<Tensor> ret_non_ess; 
    std::vector<Tensor> ret_ess;
    simplex_dimension = simplex_dimension.unsqueeze(1);    

    auto range = empty_like(pivots);
    TensorUtils::fill_range_cuda_(range); 

    auto pool_for_barcodes_non_essential = cat({pivots, range}, 1);
    auto mask_pivot = pivots.ge(0);
    
    // all dimenions mask non essential ... 
    auto mask_non_essential = mask_pivot.expand({-1, 2});

    // all dimensions mask essential ...
    auto mask_no_pivot = pivots.le(-1); 
    auto mask_rows_with_no_lowest_one = ones_like(mask_no_pivot);
    auto row_indices_with_lowest_one = pivots.masked_select(mask_pivot).toType(ScalarType::Long);

    mask_rows_with_no_lowest_one.index_fill_(0, row_indices_with_lowest_one, 0);

    auto mask_ess = mask_no_pivot.__and__(mask_rows_with_no_lowest_one);

    for (int dim=0; dim <= max_dimension; dim++){
      
      // non essentials ...
      auto mask_dim = simplex_dimension.eq(dim + 1);
      auto mask_non_essential_dim = mask_non_essential.__and__(mask_dim.expand({-1, 2}));
      auto barcodes_non_essential_dim = pool_for_barcodes_non_essential.masked_select(mask_non_essential_dim).view({-1, 2});
      
      ret_non_ess.push_back(barcodes_non_essential_dim);

      // essentials ...
      auto mask_dim_ess = simplex_dimension.eq(dim);
      auto mask_essential_dim = mask_ess.__and__(mask_dim_ess); 
      auto barcode_birth_times_essential_dim = range.masked_select(mask_essential_dim).view({-1, 1});

      ret_ess.push_back(barcode_birth_times_essential_dim);
    } 

    return std::vector<std::vector<Tensor> >({ret_non_ess, ret_ess});
  }


#pragma endregion 


std::vector<std::vector<Tensor> > calculate_persistence(   
    Tensor comp_desc_sort_ba, 
    Tensor ind_not_reduced, //TODO rename parameter accordingly to python binding 
    Tensor simplex_dimension,
    int max_dimension,
    int max_pairs = -1
    ) {

  CHECK_INPUT(comp_desc_sort_ba);
  assert(comp_desc_sort_ba.type().scalarType() == ScalarType::Int);
  CHECK_INPUT(ind_not_reduced);
  assert(ind_not_reduced.type().scalarType() == ScalarType::Long);
  CHECK_INPUT(simplex_dimension);
  assert(simplex_dimension.type().scalarType() == ScalarType::Int);

  assert(comp_desc_sort_ba.size(0) == ind_not_reduced.size(0));
  assert(ind_not_reduced.ndimension() == 1);
  assert(simplex_dimension.ndimension() == 1);
  
  int iterations = 0;

  // auto ind_not_reduced = comp_desc_sort_ba.type()
  //   .toScalarType(ScalarType::Long).tensor({simplex_dimension.size(0), 1});
  // fill_range_cuda_(ind_not_reduced);
  
  // auto tmp_pivots = comp_desc_sort_ba.slice(1, 0, 1).contiguous();
  auto scalar_0 = comp_desc_sort_ba.type().scalarTensor(0);

  // Tensor mask_not_reduced = tmp_pivots.ge(scalar_0);

  // ind_not_reduced = ind_not_reduced.masked_select(mask_not_reduced).contiguous();

  // comp_desc_sort_ba =
  //   comp_desc_sort_ba.index_select(0, ind_not_reduced).contiguous();

  Tensor mask_not_reduced, pivots, merge_pairings, new_ind_not_reduced;
  while(true){

    pivots = comp_desc_sort_ba.slice(1, 0, 1).contiguous();

    try{

      merge_pairings = find_merge_pairings(pivots, max_pairs);   

    }
    catch(NoPairsException& e){

      std::cout << "Reached end of reduction after " << iterations << " iterations" << std::endl;
      break;

    }

    comp_desc_sort_ba = merge_columns(comp_desc_sort_ba, merge_pairings);

    new_ind_not_reduced = comp_desc_sort_ba.type()
      .toScalarType(ScalarType::Long).tensor({comp_desc_sort_ba.size(0), 1});
    TensorUtils::fill_range_cuda_(new_ind_not_reduced);
    
    pivots = comp_desc_sort_ba.slice(1, 0, 1).contiguous();
    mask_not_reduced = pivots.ge(scalar_0);
    new_ind_not_reduced = new_ind_not_reduced.masked_select(mask_not_reduced).contiguous();

    comp_desc_sort_ba =
      comp_desc_sort_ba.index_select(0, new_ind_not_reduced).contiguous();
    // pivots = pivots.index_select(0, new_ind_not_reduced).contiguous();

    ind_not_reduced = ind_not_reduced.index_select(0, new_ind_not_reduced);

    iterations++;

  }

  auto real_pivots = pivots.type().tensor({simplex_dimension.size(0), 1}).fill_(
    pivots.type().scalarTensor(-1)
  );
  real_pivots.index_copy_(0, ind_not_reduced, pivots); 

  auto barcodes = read_barcodes(real_pivots, simplex_dimension, max_dimension);
  return barcodes;
}


Tensor my_test_f(Tensor t){
  auto ret = zeros_like(t);

  // my_test_kernel<<<1, 32>>>(t);

  return ret;
}


} // namespace CalcPersCuda