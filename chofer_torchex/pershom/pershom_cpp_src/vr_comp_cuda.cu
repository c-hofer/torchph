#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>

#include <vector>
#include <limits>

#include "param_checks_cuda.cuh"
#include "tensor_utils.cuh"
#include "calc_pers_cuda.cuh"
#include "vr_comp_cuda.cuh"

using namespace at;


//TODO extract in other file 
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


namespace VRCompCuda {


#pragma region binomal_table

__device__ int64_t binom_coeff(int64_t n, int64_t k){
    int64_t res = 1; 

    if ( k > n) return 0; 
 
    // Since C(n, k) = C(n, n-k)
    if ( k > n - k ){
        k = n - k;
    }
 
    // Calculate value of [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
    for (int64_t i = 0; i < k; i++)
    {
        res *= (n-i);
        res /= (i + 1);
    }
 
    return res;
}


__global__ void binomial_table_kernel(int64_t* out, int64_t max_n, int64_t max_k){
    int64_t r = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t c = blockIdx.y*blockDim.y + threadIdx.y;

    if (r < max_k && c < max_n){

        out[r*max_n + c] = binom_coeff(c, r + 1);

    }
}


/**
 * @brief 
 * 
 * @param max_n 
 * @param max_k 
 * @param type 
 * @return Tensor [max_k, max_n] where return[i, j] = binom(j, i+1)
 */
Tensor binomial_table(int64_t max_n, int64_t max_k, const Device& device){
 
    
    // auto ret = type.toScalarType(ScalarType::Long).tensor({max_k, max_n}); // TODO delete
    auto ret = at::empty({max_k, max_n}, at::dtype(at::kLong).device(device)); 

    dim3 threads_per_block = dim3(8, 8);
    dim3 num_blocks= dim3(max_k/threads_per_block.y + 1, max_n/threads_per_block.x + 1);    


    binomial_table_kernel<<<num_blocks, threads_per_block>>>(
        ret.data<int64_t>(),
        max_n, 
        max_k);      

    cudaCheckError();   

    return ret; 
}

#pragma endregion 

#pragma region write combinations table to tensor

int64_t binom_coeff_cpu(int64_t n, int64_t k){
    int64_t res = 1; 

    if ( k > n) return 0; 
 
    // Since C(n, k) = C(n, n-k)
    if ( k > n - k ){
        k = n - k;
    }
 
    // Calculate value of [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
    for (int64_t i = 0; i < k; i++)
    {
        res *= (n-i);
        res /= (i + 1);
    }
 
    return res;
}

// Unranking of a comination, c.f., https://en.wikipedia.org/wiki/Combinatorial_number_system
__device__ void unrank_combination(
    int64_t* out, // writing to this, assert length(out) == r
    const int64_t N, // rank of the combination 
    const int64_t max_n, // cobinations of elements < max_n
    const int64_t r, // number of combinations
    int64_t* const binom_table //cache of combinations, assert binom_table[i,j] == binom(j,i)
    ){

    int64_t* bt_row; 
    int64_t rest = N; 
    bool broken = false; 

    for (int64_t i=0; i<r;i++){
        bt_row = &binom_table[(r - i - 1)*max_n];

        for (int64_t j=0; j < max_n - 1; j++){
            if (bt_row[j] <= rest && bt_row[j+1] > rest){
                rest = rest - bt_row[j];
                out[i] = j;  
                broken = true;
                break; 
            }
        }

        if (!broken) {
            out[i] = max_n - 1; 
            rest = rest - bt_row[max_n -1];
        }
        
    }
}


// writes the next combination into out, e.g., out = [3, 2, 1] -> out = [4, 2, 1] 
__device__ void next_combination(int64_t* out, int64_t r){

    // If we have to increase not the first digit ... 
    for (int64_t i = 0; i < r; i++){        
        if (out[r - i - 2] > out[r - i - 1] + 1){
            out[r - i - 1] += 1;
           
            // fill the following digits with the smallest ordered sequence ... 
            for (int64_t j=0; j < i; j++){
                out[r - j - 1] = j;
            }
            return;
        }
    }

    // If the first digit has to be increased ...
    out[0] += 1;    
    // fill the following digits with the smallest ordered sequence ... 
    for (int64_t j=0; j < r - 1; j++){
            out[r - j - 1] = j;
    }
  
}


__global__ void write_combinations_to_tensor_kernel(
    int64_t* out, 
    const int64_t out_row_offset, 
    const int64_t out_row_stride, 
    const int64_t additive_constant, // additive constant which is added to the digits of each combination
    const int64_t max_n, 
    const int64_t r, 
    int64_t* binom_table, 
    const int64_t n_comb_by_thread, 
    const int64_t n_max_over_r){

    int64_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;


    if (thread_id*n_comb_by_thread < binom_coeff(max_n, r)){ // TODO use parameter instead of binom_coeff call
        int64_t* comb = new int64_t[r]; 
        unrank_combination(comb, thread_id*n_comb_by_thread, max_n, r, binom_table);

        for (int64_t i = 0; i < n_comb_by_thread; i++){

            if (thread_id*n_comb_by_thread + i >= n_max_over_r) break; 

            for (int64_t j = 0; j < r; j++){
                out[out_row_stride * (out_row_offset + thread_id*n_comb_by_thread + i ) + j] 
                    = comb[j] + additive_constant;
            }     

            next_combination(comb, r);
        }

        delete[] comb;
    }

    __syncthreads();

}

/*
Writes all combinations of {0, ... , max_n -1} of length r 
to out in lexicographical order. 
Example max_n = 4, r = 3, off_set = 1, additive_constant = 0

     out     ->      out 

-1 -1 -1 -1     -1 -1 -1 -1
-1 -1 -1 -1      2  1  0 -1 
-1 -1 -1 -1      3  1  0 -1
-1 -1 -1 -1      3  2  0 -1
-1 -1 -1 -1      3  2  1 -1
-1 -1 -1 -1     -1 -1 -1 -1
*/ 
void write_combinations_table_to_tensor(
    const Tensor& out, 
    const int64_t out_row_offset, // the 
    const int64_t additive_constant, // constant added to each digit of the combination
    const int64_t max_n, 
    const int64_t r
    ){

    CHECK_SMALLER_EQ(r, max_n); 
    const int64_t n_max_over_r = binom_coeff_cpu(max_n, r);

    CHECK_SMALLER_EQ(0, out_row_offset);
    CHECK_SMALLER_EQ(n_max_over_r + out_row_offset, out.size(0));
    CHECK_SMALLER_EQ(r, out.size(1));  
    CHECK_EQUAL(out.ndimension(), 2);


    const auto bt = binomial_table(max_n, r, out.device());
    const int n_comb_by_thread = 100; //TODO optimize
    int threads_per_block = 64; //TODO optimize

    int blocks = n_max_over_r/(threads_per_block*n_comb_by_thread) + 1;

    write_combinations_to_tensor_kernel<<<blocks, threads_per_block>>>(
        out.data<int64_t>(), 
        out_row_offset, 
        out.size(1), 
        additive_constant, 
        max_n, 
        r, 
        bt.data<int64_t>(),
        n_comb_by_thread, 
        n_max_over_r
    );

    cudaCheckError(); 
}

#pragma endregion 


Tensor l1_norm_distance_matrix(const Tensor& points){
    Tensor ret = points.unsqueeze(1).expand({points.size(0), points.size(0), points.size(1)});

    return (ret.transpose(0, 1) - ret).abs().sum(2); 
}


Tensor l2_norm_distance_matrix(const Tensor & points){
    auto x = points.unsqueeze(1).expand({points.size(0), points.size(0), points.size(1)});
    return (x.transpose(0,1) - x).pow(2).sum(2).sqrt(); 
}

#pragma region old_stuff
// std::tuple<Tensor, Tensor> get_boundary_and_filtration_info_dim_1(
//     const Tensor & point_cloud, 
//     double max_ball_radius){

//     Tensor ba_dim_1, filt_val_vec_dim_1; 
//     auto n_edges = binom_coeff_cpu(point_cloud.size(0), 2); 
//     ba_dim_1 = point_cloud.type().toScalarType(ScalarType::Long).tensor({n_edges, 2}); 

//     write_combinations_table_to_tensor(ba_dim_1, 0, 0, point_cloud.size(0)/*=max_n*/, 2/*=r*/);

//     auto distance_matrix = l1_norm_distance_matrix(point_cloud); 

//     cudaStreamSynchronize(0); // ensure that write_combinations_table_to_tensor call has finished
//     // building the vector containing the filtraiton values of the edges 
//     // in the same order as they appear in ba_dim_1...
//     auto x_indices = ba_dim_1.slice(1, 0, 1).squeeze(); 
//     auto y_indices = ba_dim_1.slice(1, 1, 2); 

//     // filling filtration vector with edge filtration values ... 
//     filt_val_vec_dim_1 = distance_matrix.index_select(0, x_indices);
//     filt_val_vec_dim_1 = filt_val_vec_dim_1.gather(1, y_indices);
//     filt_val_vec_dim_1 = filt_val_vec_dim_1.squeeze(); // 

//     // reduce to edges with filtration value <= max_ball_radius...
//     if (max_ball_radius > 0){
//         auto i_select = filt_val_vec_dim_1.le(point_cloud.type().scalarTensor(max_ball_radius)).nonzero().squeeze(); 
//         if (i_select.numel() ==  0){
//             ba_dim_1 = ba_dim_1.type().tensor({0});
//             filt_val_vec_dim_1 = filt_val_vec_dim_1.type().tensor({0}); 
//         }
//         else{
//             ba_dim_1 = ba_dim_1.index_select(0, i_select);
//             filt_val_vec_dim_1 = filt_val_vec_dim_1.index_select(0, i_select); 
//         }
//     }

//     return std::make_tuple(ba_dim_1, filt_val_vec_dim_1);
// }


// std::tuple<Tensor, Tensor> get_boundary_and_filtration_info(
//     const Tensor & filt_vals_prev_dim, 
//     int64_t dim){

//     auto n_dim_min_one_simplices = filt_vals_prev_dim.size(0); 

//     Tensor new_boundary_info, new_filt_vals;

//     if (n_dim_min_one_simplices < dim + 1){
//         // There are not enough dim-1 simplices ...
//         new_boundary_info = filt_vals_prev_dim.type().toScalarType(ScalarType::Long).tensor({0, dim + 1});
//         new_filt_vals = filt_vals_prev_dim.type().tensor({0});
//     }
//     else{
//         // There are enough dim-1 simplices ...
//         auto n_new_simplices = binom_coeff_cpu(n_dim_min_one_simplices, dim + 1); 
//         auto n_simplices_prev_dim = filt_vals_prev_dim.size(0); 

//         new_boundary_info = filt_vals_prev_dim.type().toScalarType(ScalarType::Long).tensor({n_new_simplices, dim + 1}); 

//         // write combinations ... 
//         write_combinations_table_to_tensor(new_boundary_info, 0, 0, n_simplices_prev_dim, dim + 1); 
//         cudaStreamSynchronize(0); 

//         auto bi_cloned = new_boundary_info.clone(); // we have to clone here other wise auto-grad does not work!
//         new_filt_vals = filt_vals_prev_dim.expand({n_new_simplices, filt_vals_prev_dim.size(0)});
//         new_filt_vals = new_filt_vals.gather(1, bi_cloned); 
//         new_filt_vals = std::get<0>(new_filt_vals.max(1));

//         // If we have just one simplex of the current dimension this
//         // condition avoids that new_filt_vals is squeezed to a 0-dim 
//         // Tensor
//         if (new_filt_vals.ndimension() != 1){      
//             new_filt_vals = new_filt_vals.squeeze(); 
//         }
//     }

//     return std::make_tuple(new_boundary_info, new_filt_vals); 
// }


// void make_simplex_ids_compatible_within_dimensions(
//     std::vector<int64_t> & n_simplices_by_dim, 
//     std::vector<std::tuple<Tensor, Tensor>> & boundary_and_filtration_by_dim
//     ){
//     auto index_offset = n_simplices_by_dim.at(0);
//     for (int i=1; i < boundary_and_filtration_by_dim.size(); i++){
//         auto boundary_info = std::get<0>(boundary_and_filtration_by_dim.at(i)); 
//         boundary_info.add_(index_offset); 

//         auto n_simplices_in_prev_dim = std::get<0>(boundary_and_filtration_by_dim.at(i-1)).size(0); 
//         index_offset += n_simplices_in_prev_dim;
//     }
// }


// Tensor get_simplex_dimension_tensor(
//     int64_t n_non_vertex_simplices, 
//     std::vector<int64_t> & n_simplices_by_dim, 
//     int64_t max_dimension, 
//     Type& type
//     ){

//     auto ret = type.tensor({n_non_vertex_simplices + n_simplices_by_dim.at(0)}); 
   
//     int64_t copy_offset = 0; 
//     for (int i = 0; i <= (max_dimension == 0 ? 1 : max_dimension) ; i++){
//         ret.slice(0, copy_offset, copy_offset + n_simplices_by_dim.at(i)).fill_(i); 
//         copy_offset += n_simplices_by_dim.at(i); 
//     }

//     return ret; 
// }


// Tensor get_filtrations_values_vector(
//     std::vector<std::tuple<Tensor, Tensor>> & boundary_and_filtration_by_dim
//     ){
    
//     std::vector<Tensor> filt_values_non_vertex_simplices; 
//     for (int i = 0; i < boundary_and_filtration_by_dim.size(); i++){
    
//         auto filt_vals = std::get<1>(boundary_and_filtration_by_dim.at(i));
//         filt_values_non_vertex_simplices.push_back(filt_vals);  
//     } 

//     return cat(filt_values_non_vertex_simplices, 0);     
// }


// std::vector<Tensor> get_calculate_persistence_args_simplices_only(
//     int64_t n_vertices, 
//     int64_t max_dimension,
//     Type & ba_type, 
//     const Tensor & point_cloud
//     ){

//     std::vector<Tensor> ret; 
//     ret.push_back(ba_type.tensor({0, 2*(max_dimension + 1)}));
//     ret.push_back(ba_type.tensor({0}));
//     ret.push_back(ba_type.zeros({n_vertices}));

//     // We generate the 0-vector in this way to ensure that point_cloud
//     // will have zero gradients instead of None after a backward call
//     // in pytorch ... 
//     auto filtration_values = point_cloud.slice(1, 0, 1).squeeze().clone();
//     filtration_values.fill_(0); 
//     ret.push_back(filtration_values);

//     return ret; 
// }


// //TODO refactor 
// std::vector<Tensor> vr_l1_generate_calculate_persistence_args(
//     const Tensor& point_cloud,
//     int64_t max_dimension, 
//     double max_ball_radius
//     ){

//     CHECK_TENSOR_CUDA_CONTIGUOUS(point_cloud);
//     CHECK_SMALLER_EQ(max_dimension + 1, point_cloud.size(0)); 
//     CHECK_SMALLER_EQ(0, max_ball_radius);


//     std::vector<Tensor> ret;
//     Type& Long = point_cloud.type().toScalarType(ScalarType::Long);

//     // 1. generate boundaries and filtration values ...
//     // boundary_and_filtration_info_by_dim[i] == (enumerated boundary combinations, filtration values) of 
//     // dimension i + 1. 
//     std::vector<std::tuple<Tensor, Tensor>> boundary_and_filtration_by_dim;

//     boundary_and_filtration_by_dim.push_back(
//         get_boundary_and_filtration_info_dim_1(point_cloud, max_ball_radius)
//     );

//     for (int dim = 2; dim <= max_dimension; dim++){
//         auto filt_vals_prev_dim = std::get<1>(boundary_and_filtration_by_dim.at(dim - 2));

//         boundary_and_filtration_by_dim.push_back(
//             get_boundary_and_filtration_info(filt_vals_prev_dim, dim)
//         );
//     }

//     // 2. Create helper variables which contain meta info about simplex numbers ... 
//     int64_t n_non_vertex_simplices = 0;
//     int64_t n_simplices = point_cloud.size(0); 
//     std::vector<int64_t> n_simplices_by_dim; 
//     n_simplices_by_dim.push_back(point_cloud.size(0)); 

//     for (int i = 0; i < boundary_and_filtration_by_dim.size(); i++){
//         auto boundary_info = std::get<0>(boundary_and_filtration_by_dim.at(i));
//         n_non_vertex_simplices += boundary_info.size(0); 
//         n_simplices += boundary_info.size(0); 
//         n_simplices_by_dim.push_back(boundary_info.size(0)); 
//     }

//     // TODO returning in mid of function is not nice. Can we improve this? 
//     // If there are only vertices, we omit the rest of the function and 
//     // return the arguments for calculate_persistence directly ... 
//     if (n_non_vertex_simplices == 0){
//         return get_calculate_persistence_args_simplices_only(
//             n_simplices_by_dim.at(0), 
//             max_dimension, 
//             Long, 
//             point_cloud
//         );
//     }

//     // 3. Make simplex id's compatible within dimensions ... 
//     /*    
//     In order to keep indices in the boundary info tensors 
//     compatible within dimensions we have to add an offset
//     to the enumerated combinations, starting with 
//     dimension 2 simplices (the boundaries of dim 1 simplices are vertices, 
//     hence the enumeration of the boundary combinations is valid)
//     */
//     make_simplex_ids_compatible_within_dimensions(n_simplices_by_dim, boundary_and_filtration_by_dim);

//     // 4. Create simplex_dimension vector ... 
//     auto simplex_dimension = get_simplex_dimension_tensor(n_non_vertex_simplices, n_simplices_by_dim, max_dimension, Long);

//     // 5. Create filtration vector ... 
//     Tensor filtration_values_vector = get_filtrations_values_vector(boundary_and_filtration_by_dim);
       

//     /* 
//     This is a dirty hack to ensure that simplices do not occour before their boundaries 
//     in the filtration. As the filtration is raised to higher dimensional simplices by 
//     taking the maxium of the involved edge filtration values and sorting does not guarantee
//     a specific ordering in case of equal values we are forced to ensure a well defined 
//     filtration by adding an increasing epsilon to each dimension. Later this has to be 
//     substracted again. 
//     Example: f([1,2,3]) = max(f([1,2]), f([3,1]), f([2,3])) --> w.l.o.g. f([1,2,3]) == f([1,2])
//     Hence we set f([1,2,3]) = f([1,2]) + epsilon
//     */
//     auto filt_add_hack_values = filtration_values_vector.type().tensor({filtration_values_vector.size(0)}).fill_(0);
    
//     {
//         if (max_dimension >= 2 && n_simplices_by_dim.at(2) > 0){
            
//             // we take epsilon of float to ensure that it is well defined even if 
//             // we decide to alter the floating point type of the filtration values 
//             // realm 
//             float add_const_base_value = 100 * std::numeric_limits<float>::epsilon(); // multily with 100 to be save against rounding issues
//             auto copy_offset = n_simplices_by_dim.at(1); 

//             for (int dim = 2; dim <= max_dimension; dim++){
//                 filt_add_hack_values.slice(0, copy_offset, copy_offset + n_simplices_by_dim.at(dim))
//                     .fill_(add_const_base_value); 

//                 add_const_base_value += add_const_base_value; 
//                 copy_offset += n_simplices_by_dim.at(dim); 
//             }

//             filtration_values_vector += filt_add_hack_values;
//         }

//         filt_add_hack_values = filt_add_hack_values.clone();
    
//     }

//     //6 Do sorting ...
    
//     auto sort_filt_res = filtration_values_vector.sort(0);
//     auto sorted_filtration_values_vector = std::get<0>(sort_filt_res);
//     auto sort_i_filt = std::get<1>(sort_filt_res); 

//     // revert filtration hack if necessary ...
//     if (max_dimension >= 2 && n_simplices_by_dim.at(2) > 0){
//         filt_add_hack_values = filt_add_hack_values.index_select(0, sort_i_filt); 
//         sorted_filtration_values_vector -= filt_add_hack_values;
//     }
//     // now the filtration is cleaned and we can continue. 

//     // Simplex ids in boundary_array entries include vertices.
//     // As filtration_value_vector so far starts with edges we have to take care of this. 
//     auto dim_0_filt_values = sorted_filtration_values_vector.type().zeros({n_simplices_by_dim.at(0)}); 
//     sorted_filtration_values_vector = cat({dim_0_filt_values, sorted_filtration_values_vector}, 0); 
  

//     // Sort simplex_dimension ...
//     simplex_dimension.slice(0, n_simplices_by_dim.at(0)) = 
//         simplex_dimension.slice(0, n_simplices_by_dim.at(0)).index_select(0, sort_i_filt);

//     // Copy boundary_info of each dimension into the final boundary array ... 
//     auto boundary_array = point_cloud.type().toScalarType(ScalarType::Long)
//         .tensor({n_non_vertex_simplices, 
//                  2*((max_dimension == 0 ? 1:max_dimension) + 1)});

//     {
//         boundary_array.fill_(-1); 

//         // copy edges ... 
//         auto edge_boundary_info = std::get<0>(boundary_and_filtration_by_dim.at(0));
//         boundary_array.slice(0, 0, n_simplices_by_dim.at(1)).slice(1, 0, 2) = edge_boundary_info; 

//         // copy higher dimensional simplices
//         if (max_dimension >= 2){
//             // we need a look up table which lets us change the simplex ids we get from the initial 
//             // enumeration (write_combinations_table_to_tensor) to the id the have w.r.t. the ordering
//             // of the filtration values. We create this table now ...

//             // This gives us the mapping id -> new_id w.r.t. sorting by filtration values ...
//             auto look_up_table_row = std::get<1>(sort_i_filt.sort(0));

//             // look_up_table_row is yet based on id's without vertices, we adapt theis now ...
//             auto dummy_sort_indices = sort_i_filt.type().tensor({n_simplices_by_dim.at(0)}).fill_(std::numeric_limits<int64_t>::max());
//             look_up_table_row = look_up_table_row + n_simplices_by_dim.at(0); 

//             // as vertices have no boundary we will never select a value of the first #vertices entries, 
//             // but we need look_up_table_row.size(0) == #simplices in order to get a consistent mapping...
//             look_up_table_row = cat({dummy_sort_indices, look_up_table_row}, 0); 

//             int64_t copy_offset = n_simplices_by_dim.at(1);             

//             for (int i = 1; i < max_dimension; i++){

//                 auto boundary_info = std::get<0>(boundary_and_filtration_by_dim.at(i)); 

//                 if (boundary_info.size(0) == 0){
//                     continue; 
//                 }

//                 auto look_up_table = look_up_table_row.expand({boundary_info.size(0), look_up_table_row.size(0)});  

//                 // Apply ordering to row content ... 
//                 boundary_info = look_up_table.gather(1, boundary_info); 

//                 // Apply ordering to rows ...
//                 boundary_info = std::get<0>(boundary_info.sort(1, /*descending=*/true));

//                 boundary_array.slice(0, copy_offset, copy_offset + boundary_info.size(0)).slice(1, 0, boundary_info.size(1))
//                      = boundary_info; 

//                 copy_offset += boundary_info.size(0); 
//             }
//         }
//     }

//     // Sort boundary_array rows ...
//     boundary_array = boundary_array.index_select(0, sort_i_filt);  

//     //7. generate ba_row_i_to_bm_col_i
//     auto ba_row_i_to_bm_col_i = boundary_array.type().tensor({boundary_array.size(0)});
//     TensorUtils::fill_range_cuda_(ba_row_i_to_bm_col_i); 
//     ba_row_i_to_bm_col_i += n_simplices_by_dim.at(0); 

//     //8. returning ... 
//     ret.push_back(boundary_array); 
//     ret.push_back(ba_row_i_to_bm_col_i);
//     ret.push_back(simplex_dimension); 
//     ret.push_back(sorted_filtration_values_vector);  

//     return ret;
// }
#pragma endregion


#pragma region co-faces from combinations


__global__ void mask_of_valid_co_faces_from_combinations_kernel(
    int64_t* combinations, 
    int64_t combinations_size_0,
    int64_t combinations_size_1,
    int64_t* faces, 
    int64_t faces_size_1,
    int64_t* out
    ){
    
    int64_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;

    if (thread_id < combinations_size_0){

        const int n_faces = combinations_size_1*(combinations_size_1-1);
        int64_t* current_comb = combinations + thread_id*combinations_size_1; 
        int64_t* tmp = new int64_t[n_faces + 1];

        tmp[0] = -1;
        for (int i = 0; i < combinations_size_1; i++){
            for(int j = 0; j < faces_size_1; j++){
                tmp[(i*faces_size_1 + j) + 1] = 
                faces[current_comb[i]*faces_size_1 + j];
            }
        }


        thrust::device_ptr<int64_t> dptr_tmp(tmp);
        thrust::sort(thrust::seq, dptr_tmp, dptr_tmp + n_faces+1);

        bool is_boundary = true; 
        for (int i = 1; i < n_faces+1; i += 2){
            is_boundary = is_boundary && (dptr_tmp[i] == (dptr_tmp[i+1]));
            is_boundary = is_boundary && (dptr_tmp[i-1] < dptr_tmp[i]); 
        }

        out[thread_id] = is_boundary ? 1 : 0; 

        delete[] tmp;  
    }
} 


Tensor co_faces_from_combinations(
    const Tensor & combinations, 
    const Tensor & faces
    ){
    // auto mask = combinations.type().toScalarType(ScalarType::Long)
    //                         .tensor({combinations.size(0)}); TODO delete
    auto mask = at::empty(
        {combinations.size(0)}, 
        at::dtype(at::kLong).device(combinations.device()));

    mask.fill_(0); 

    int threads_per_block = 64; //TODO optimize
    int blocks = combinations.size(0)/threads_per_block + 1;

    mask_of_valid_co_faces_from_combinations_kernel<<<blocks, threads_per_block>>>(
        combinations.data<int64_t>(), 
        combinations.size(0),
        combinations.size(1),
        faces.data<int64_t>(),
        faces.size(1), 
        mask.data<int64_t>()
    );
    
    cudaStreamSynchronize(0); 
    cudaCheckError(); 

    auto indices = mask.nonzero().squeeze(); 
    return combinations.index_select(0, indices); 
}


#pragma endregion

PointCloud2VR PointCloud2VR_factory(const std::string & distance){
    if (distance.compare("l1") == 0) return PointCloud2VR(&l1_norm_distance_matrix); 
    if (distance.compare("l2") == 0) return PointCloud2VR(&l2_norm_distance_matrix); 
    
    throw std::range_error("Expected 'l1' or 'l2'!");
    
}


void PointCloud2VR::init_state(
    const Tensor & point_cloud, 
    int64_t max_dimension, 
    double max_ball_radius
    ){
        CHECK_TENSOR_CUDA_CONTIGUOUS(point_cloud);
        CHECK_SMALLER_EQ(max_dimension + 1, point_cloud.size(0)); 
        CHECK_SMALLER_EQ(0, max_ball_radius);

        // this->RealType = &point_cloud.type(); TODO delete
        this->tensopt_real = at::TensorOptions()
            .dtype(point_cloud.dtype())
            .device(point_cloud.device());  

        // this->IntegerType = &point_cloud.type().toScalarType(this->IntegerScalarType); TODO delete
        this ->tensopt_int = at::TensorOptions()
            .dtype(at::kLong)
            .device(point_cloud.device());

        this->point_cloud = point_cloud;
        this->max_dimension = max_dimension;
        this->max_ball_radius = max_ball_radius; 

        this->n_simplices_by_dim.push_back(point_cloud.size(0));
        // this->filtration_values_by_dim.push_back(
        //     this->RealType->tensor({point_cloud.size(0)}).fill_(0)
        //     ); TODO delete
        this->filtration_values_by_dim.push_back(
            at::empty({point_cloud.size(0)}, 
            this->tensopt_real)
            .fill_(0)
            ); 
}


void PointCloud2VR::make_boundary_info_edges(){
    Tensor ba_dim_1, filt_val_vec_dim_1; 
    auto n_edges = binom_coeff_cpu(point_cloud.size(0), 2); 
    // ba_dim_1 = this->IntegerType->tensor({n_edges, 2}); TODO delete
    ba_dim_1 = at::empty({n_edges, 2}, this->tensopt_int); 

    write_combinations_table_to_tensor(ba_dim_1, 0, 0, point_cloud.size(0)/*=max_n*/, 2/*=r*/);

    auto distance_matrix = this->get_distance_matrix(point_cloud); 

    cudaStreamSynchronize(0); // ensure that write_combinations_table_to_tensor call has finished
    // building the vector containing the filtraiton values of the edges 
    // in the same order as they appear in ba_dim_1...
    auto x_indices = ba_dim_1.slice(1, 0, 1).squeeze(); 
    auto y_indices = ba_dim_1.slice(1, 1, 2); 

    // filling filtration vector with edge filtration values ... 
    filt_val_vec_dim_1 = distance_matrix.index_select(0, x_indices);
    filt_val_vec_dim_1 = filt_val_vec_dim_1.gather(1, y_indices);
    filt_val_vec_dim_1 = filt_val_vec_dim_1.squeeze(); // 

    // reduce to edges with filtration value <= max_ball_radius...
    if (max_ball_radius > 0){
        auto i_select = filt_val_vec_dim_1.le(point_cloud.type().scalarTensor(max_ball_radius)).nonzero().squeeze(); 
        if (i_select.numel() ==  0){

            // ba_dim_1 = ba_dim_1.type().tensor({0}); TODO delete 
            ba_dim_1 = at::empty({0}, ba_dim_1.options()); 

            // filt_val_vec_dim_1 = filt_val_vec_dim_1.type().tensor({0}); TODO delete
            filt_val_vec_dim_1 = at::empty({0}, filt_val_vec_dim_1.options()); 
        }
        else{
            ba_dim_1 = ba_dim_1.index_select(0, i_select);
            filt_val_vec_dim_1 = filt_val_vec_dim_1.index_select(0, i_select); 
        }
    }

    this->boundary_info_non_vertices.push_back(ba_dim_1);
    this->filtration_values_by_dim.push_back(filt_val_vec_dim_1); 
    this->n_simplices_by_dim.push_back(ba_dim_1.size(0)); 
}


void PointCloud2VR::make_boundary_info_non_edges(){


    Tensor filt_vals_prev_dim;
    int64_t n_dim_min_one_simplices; 
    Tensor new_boundary_info, new_filt_vals;

    for (int dim = 2; dim <= this->max_dimension; dim++){

        filt_vals_prev_dim = this->filtration_values_by_dim.at(dim - 1);
        n_dim_min_one_simplices = filt_vals_prev_dim.size(0); 

        if (n_dim_min_one_simplices < dim + 1){
            // There are not enough dim-1 simplices ...
            // new_boundary_info = filt_vals_prev_dim.type().toScalarType(ScalarType::Long).tensor({0, dim + 1}); TODO delete
            new_boundary_info = at::empty({0, dim + 1}, this->tensopt_int);

            // new_filt_vals = filt_vals_prev_dim.type().tensor({0});  TODO delete
            new_filt_vals = at::empty({0}, this->tensopt_real);
        }
        else{
            // There are enough dim - 1 simplices ...

            // auto combinations = filt_vals_prev_dim.type().toScalarType(ScalarType::Long).tensor(
            // {binom_coeff_cpu(n_dim_min_one_simplices, dim + 1), dim + 1}); TODO delete

            auto combinations = at::empty(
                {binom_coeff_cpu(n_dim_min_one_simplices, dim + 1), dim + 1}, 
                this->tensopt_int); 

            // write combinations ... 
            write_combinations_table_to_tensor(combinations, 0, 0, n_dim_min_one_simplices, dim + 1); 
            cudaStreamSynchronize(0); 

            new_boundary_info = co_faces_from_combinations(combinations, this->boundary_info_non_vertices.at(dim - 2)); 
            cudaStreamSynchronize(0); 
            auto bi_cloned = new_boundary_info.clone(); // we have to clone here other wise auto-grad does not work!
            new_filt_vals = filt_vals_prev_dim.expand({new_boundary_info.size(0), filt_vals_prev_dim.size(0)});

            new_filt_vals = new_filt_vals.gather(1, bi_cloned); 
            new_filt_vals = std::get<0>(new_filt_vals.max(1));

            // If we have just one simplex of the current dimension this
            // condition avoids that new_filt_vals is squeezed to a 0-dim 
            // Tensor
            if (new_filt_vals.ndimension() != 1){      
                new_filt_vals = new_filt_vals.squeeze(); 
            }
        }
        this->n_simplices_by_dim.push_back(new_boundary_info.size(0)); 
        this->boundary_info_non_vertices.push_back(new_boundary_info);
        this->filtration_values_by_dim.push_back(new_filt_vals);

    }
}


void PointCloud2VR::make_simplex_ids_compatible_within_dimensions(){
    
    auto index_offset = this->n_simplices_by_dim.at(0);
    int dim; 
    for (int i=1; i < this->boundary_info_non_vertices.size(); i++){

        dim = i + 1;
        auto boundary_info = this->boundary_info_non_vertices.at(i); 
        boundary_info.add_(index_offset); 
        
        index_offset += this->n_simplices_by_dim.at(dim-1);
    }
}


void PointCloud2VR::make_simplex_dimension_vector(){
    int64_t n_simplices = 0;
    for (int i = 0; i < this->n_simplices_by_dim.size(); i++){
        n_simplices += this->n_simplices_by_dim.at(i); 
    }

    // simplex_dimension_vector = this->IntegerType->tensor({n_simplices}); TODO delete
    simplex_dimension_vector = at::empty({n_simplices}, 
                                         this->tensopt_int); 

    auto max_dimension = this->max_dimension; 
   
    int64_t copy_offset = 0; 
    for (int i = 0; i <= (max_dimension == 0 ? 1 : max_dimension) ; i++){
        simplex_dimension_vector
            .slice(0, copy_offset, copy_offset + this->n_simplices_by_dim.at(i))
            .fill_(i); 

        copy_offset += n_simplices_by_dim.at(i); 
    }

    this->simplex_dimension_vector = simplex_dimension_vector; 
}


void PointCloud2VR::make_filtration_values_vector_without_vertices(){
    
    std::vector<Tensor> filt_values_non_vertex_simplices; 
    for (int i = 1; i < this->filtration_values_by_dim.size(); i++){
    
        auto filt_vals = this->filtration_values_by_dim.at(i); 
        filt_values_non_vertex_simplices.push_back(filt_vals);  
    } 

    this->filtration_values_vector_without_vertices = cat(filt_values_non_vertex_simplices, 0);     
}


void PointCloud2VR::do_filtration_add_eps_hack(){
    /* 
    This is a dirty hack to ensure that simplices do not occour before their boundaries 
    in the filtration. As the filtration is raised to higher dimensional simplices by 
    taking the maxium of the involved edge filtration values and sorting does not guarantee
    a specific ordering in case of equal values we are forced to ensure a well defined 
    filtration by adding an increasing epsilon to each dimension. Later this has to be 
    substracted again. 
    Example: f([1,2,3]) = max(f([1,2]), f([3,1]), f([2,3])) --> w.l.o.g. f([1,2,3]) == f([1,2])
    Hence we set f([1,2,3]) = f([1,2]) + epsilon
    */    
    if (this->max_dimension >= 2 && this->n_simplices_by_dim.at(2) > 0){

        // auto filt_add_hack_values = this->RealType->tensor(
        //     {this->filtration_values_vector_without_vertices.size(0)}).fill_(0); TODO delete
        auto filt_add_hack_values = at::empty(
            {this->filtration_values_vector_without_vertices.size(0)},
            this->tensopt_real)
            .fill_(0);
        
        // we take epsilon of float to ensure that it is well defined even if 
        // we decide to alter the floating point type of the filtration values 
        // realm 
        float add_const_base_value = 100 * std::numeric_limits<float>::epsilon(); // multily with 100 to be save against rounding issues
        auto copy_offset = this->n_simplices_by_dim.at(1); 

        for (int dim = 2; dim <= max_dimension; dim++){
            filt_add_hack_values.slice(0, copy_offset, copy_offset + this->n_simplices_by_dim.at(dim))
                .fill_(add_const_base_value); 

            add_const_base_value += add_const_base_value; 
            copy_offset += this->n_simplices_by_dim.at(dim); 
        }

        this->filtration_values_vector_without_vertices += filt_add_hack_values;
        this->filtration_add_eps_hack_values = filt_add_hack_values;
    }

}


void PointCloud2VR::make_sorting_infrastructure(){
    auto sort_ret = this->filtration_values_vector_without_vertices.sort(0);     
    this->sort_indices_without_vertices = std::get<1>(sort_ret);
    this->sort_indices_without_vertices_inverse = 
        std::get<1>(this->sort_indices_without_vertices.sort(0));
}


void PointCloud2VR::undo_filtration_add_eps_hack(){
    if (this->max_dimension >= 2 && this->n_simplices_by_dim.at(2) > 0){
        this->filtration_values_vector_without_vertices -=
            this->filtration_add_eps_hack_values; 
    }
}


void PointCloud2VR::make_sorted_filtration_values_vector(){
    // auto dim_0_filt_values = this->RealType->empty({n_simplices_by_dim.at(0)}); TODO delete
    auto dim_0_filt_values = at::empty({n_simplices_by_dim.at(0)}, this->tensopt_real);
    
    dim_0_filt_values.fill_(0);

    auto tmp = this->filtration_values_vector_without_vertices
        .index_select(0, this->sort_indices_without_vertices);

    tmp = cat({dim_0_filt_values, tmp}); 

    this->sorted_filtration_values_vector = tmp;  
}


void PointCloud2VR::make_boundary_array_rows_unsorted(){
    auto n_non_vertex_simplices = 0;
    for (int i=1; i < this->n_simplices_by_dim.size(); i++){
        n_non_vertex_simplices += this->n_simplices_by_dim.at(i); 
    }

    auto ba = at::empty(
        {n_non_vertex_simplices, (this->max_dimension == 0 ? 4 : (max_dimension + 1)*2)},
        this->tensopt_int);
    ba.fill_(-1);

    // copy edges ...
    auto edge_boundary_info = this->boundary_info_non_vertices.at(0);
    ba.slice(0, 0, this->n_simplices_by_dim.at(1)).slice(1, 0, 2)
         = edge_boundary_info; 

    if (this->max_dimension >= 2){ 

        auto look_up_table_row = this->sort_indices_without_vertices_inverse;

        // adapt to indexation with vertices ... 
        look_up_table_row += this->n_simplices_by_dim.at(0); 

        // ensure length is according to indexation with vertices ...
        auto dummy_vals = 
            at::empty({n_simplices_by_dim.at(0)}, look_up_table_row.options())
            .fill_(std::numeric_limits<int64_t>::max());

        look_up_table_row = cat({dummy_vals, look_up_table_row}, 0);

        int64_t copy_offset = this->n_simplices_by_dim.at(1);  


        for (int dim = 2; dim <= this->max_dimension; dim++){

            auto boundary_info = this->boundary_info_non_vertices.at(dim-1); 

            if (boundary_info.size(0) == 0){
                        continue; 
            }
            
            auto look_up_table = look_up_table_row.expand(
                {boundary_info.size(0), look_up_table_row.size(0)});  

            // Apply ordering to row content ... 
            boundary_info = look_up_table.gather(1, boundary_info); 

            // Ensure row contents are descendingly ordered ...
            boundary_info = std::get<0>(boundary_info.sort(1, /*descending=*/true));

            ba.slice(0, copy_offset, copy_offset + boundary_info.size(0))
                        .slice(1, 0, boundary_info.size(1))
                = boundary_info; 

            copy_offset += boundary_info.size(0);
        }
    }

    this->boundary_array = ba; 
}


void PointCloud2VR::apply_sorting_to_rows(){
    this->boundary_array = boundary_array.index_select(
        0,
        this->sort_indices_without_vertices
    );

    auto simp_dim_slice = this->simplex_dimension_vector.slice(0, this->n_simplices_by_dim.at(0));
    
    simp_dim_slice.copy_(simp_dim_slice.index_select(0, this->sort_indices_without_vertices)); 
}


void PointCloud2VR::make_ba_row_i_to_bm_col_i_vector(){
    auto tmp = at::empty({this->boundary_array.size(0)}, this->tensopt_int);
    TensorUtils::fill_range_cuda_(tmp); 
    tmp += this->n_simplices_by_dim.at(0); 

    this->ba_row_i_to_bm_col_i_vector = tmp; 
}

std::vector<Tensor> PointCloud2VR::operator()(
    const Tensor & point_cloud, 
    int64_t max_dimension, 
    double max_ball_radius){
    
    std::vector<Tensor> ret; 

    this->init_state(point_cloud, max_dimension, max_ball_radius); 
    this->make_boundary_info_edges(); 

    if (this->n_simplices_by_dim.at(1) > 0){
        this->make_boundary_info_non_edges(); 
        this->make_simplex_ids_compatible_within_dimensions(); 
        this->make_simplex_dimension_vector(); 
        this->make_filtration_values_vector_without_vertices(); 
        this->do_filtration_add_eps_hack(); 
        this->make_sorting_infrastructure(); 
        this->undo_filtration_add_eps_hack(); 
        this->make_sorted_filtration_values_vector();
        this->make_boundary_array_rows_unsorted(); 
        this->apply_sorting_to_rows(); 
        this->make_ba_row_i_to_bm_col_i_vector();


        ret.push_back(this->boundary_array);
        ret.push_back(this->ba_row_i_to_bm_col_i_vector);
        ret.push_back(this->simplex_dimension_vector);
        ret.push_back(this->sorted_filtration_values_vector);
    }
    else
    {
        ret.push_back(at::empty({0, 2*(max_dimension + 1)}, this->tensopt_int));
        ret.push_back(at::empty({0}, this->tensopt_int));
        ret.push_back(at::zeros({point_cloud.size(0)}, this->tensopt_int));

        // We generate the 0-vector in this way to ensure that point_cloud
        // will have zero gradients instead of None after a backward call
        // in pytorch ... 
        auto filtration_values = point_cloud.slice(1, 0, 1).squeeze().clone();
        filtration_values.fill_(0); 
        ret.push_back(filtration_values);
    }
    return ret; 
}


std::vector<std::vector<Tensor>> calculate_persistence_output_to_barcode_tensors(
    const std::vector<std::vector<Tensor>>& calculate_persistence_output,
    const Tensor & filtration_values){
    std::vector<std::vector<Tensor>> ret; 

    std::vector<Tensor> non_essential_barcodes; 
    {
        auto non_essentials = calculate_persistence_output.at(0);
        Tensor birth_death_i, births, birth_i, deaths, death_i, barcodes, i_birth_ne_death; 
        for (int i = 0; i < non_essentials.size(); i++){

            birth_death_i = non_essentials.at(i); 

            if(birth_death_i.numel() == 0){
                barcodes = at::empty({0, 2}, filtration_values.options()); 
            }
            else {
                birth_i = birth_death_i.slice(1, 0, 1).squeeze(); 
                births = filtration_values.index_select(0, birth_i);

                death_i = birth_death_i.slice(1, 1, 2).squeeze();
                deaths = filtration_values.index_select(0, death_i);
                i_birth_ne_death = births.ne(deaths).nonzero().squeeze(); 

                if (i_birth_ne_death.numel() != 0){
                    births = births.index_select(0, i_birth_ne_death);
                    deaths = deaths.index_select(0, i_birth_ne_death);
                    barcodes = stack({births, deaths}, 1); 
                }
                else{
                    barcodes = at::empty({0, 2}, filtration_values.options()); 
                }
                
            }
            non_essential_barcodes.push_back(barcodes); 
        }
        ret.push_back(non_essential_barcodes);    
    }

    std::vector<Tensor> essential_barcodes; 
    {   
        auto essentials = calculate_persistence_output.at(1); 
        Tensor birth_i, births, barcodes; 
        for (int i = 0; i < essentials.size(); i++){

            birth_i = essentials.at(i).squeeze(); 

            if (birth_i.numel() == 0){
                barcodes = at::empty({0, 1}, filtration_values.options());
            }
            else {
                barcodes = filtration_values.index_select(0, birth_i); 
            }
            
            essential_barcodes.push_back(barcodes); 
        }

        ret.push_back(essential_barcodes); 
    }

    return ret; 
}


std::vector<std::vector<Tensor>> vr_persistence(
    const Tensor& point_cloud,
    int64_t max_dimension, 
    double max_ball_radius, 
    const std::string & metric){    
    
    std::vector<std::vector<Tensor>> ret; 
    auto args_generator = PointCloud2VR_factory(metric);

    auto args = args_generator(point_cloud, max_dimension, max_ball_radius);
    
    auto pers = CalcPersCuda::calculate_persistence(
        args.at(0), args.at(1), args.at(2), max_dimension, -1
    );

    cudaStreamSynchronize(0);

    auto filtration_values = args.at(3); 
    ret = calculate_persistence_output_to_barcode_tensors(pers, filtration_values); 

    return ret;
}


} // namespace VRCompCuda 