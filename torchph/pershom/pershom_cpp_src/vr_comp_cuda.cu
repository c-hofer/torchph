#include <torch/extension.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

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
#include "cuda_checks.cuh"


using namespace torch;


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

    at::OptionalDeviceGuard guard(device);

    auto ret = torch::empty({max_k, max_n}, torch::dtype(torch::kInt64).device(device)); 

    dim3 threads_per_block = dim3(8, 8);
    dim3 num_blocks= dim3(max_k/threads_per_block.y + 1, max_n/threads_per_block.x + 1);    


    binomial_table_kernel<<<num_blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        ret.data_ptr<int64_t>(),
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

    at::OptionalDeviceGuard guard(device_of(out));

    const auto bt = binomial_table(max_n, r, out.device());
    const int n_comb_by_thread = 100; //TODO optimize
    int threads_per_block = 64; //TODO optimize

    int blocks = n_max_over_r/(threads_per_block*n_comb_by_thread) + 1;

    write_combinations_to_tensor_kernel<<<blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out.data_ptr<int64_t>(), 
        out_row_offset, 
        out.size(1), 
        additive_constant, 
        max_n, 
        r, 
        bt.data_ptr<int64_t>(),
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
        thrust::sort(thrust::seq, dptr_tmp, dptr_tmp + n_faces + 1);

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

    CHECK_SAME_DEVICE(combinations, faces); 
    at::OptionalDeviceGuard guard(device_of(combinations));

    auto mask = torch::empty(
        {combinations.size(0)}, 
        torch::dtype(torch::kInt64).device(combinations.device()));

    mask.fill_(0); 

    int threads_per_block = 64; //TODO optimize
    int blocks = combinations.size(0)/threads_per_block + 1;

    mask_of_valid_co_faces_from_combinations_kernel<<<blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        combinations.data_ptr<int64_t>(), 
        combinations.size(0),
        combinations.size(1),
        faces.data_ptr<int64_t>(),
        faces.size(1), 
        mask.data_ptr<int64_t>()
    );
    
    cudaCheckError(); 

    auto indices = mask.nonzero().squeeze(); 

    return combinations.index_select(0, indices); 
}


#pragma endregion


void VietorisRipsArgsGenerator::init_state(
    const Tensor & distance_matrix, 
    int64_t max_dimension, 
    double max_ball_diameter
    ){
        CHECK_TENSOR_CUDA_CONTIGUOUS(distance_matrix);
        CHECK_SMALLER_EQ(max_dimension + 1, distance_matrix.size(0)); 
        CHECK_SMALLER_EQ(0, max_ball_diameter);
        CHECK_EQUAL(distance_matrix.size(0), distance_matrix.size(1));

        this->tensopt_real = torch::TensorOptions()
            .dtype(distance_matrix.dtype())
            .device(distance_matrix.device());  

        this ->tensopt_int = torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(distance_matrix.device());

        this->distance_matrix = distance_matrix;
        this->max_dimension = max_dimension;
        this->max_ball_diameter = max_ball_diameter; 

        this->n_simplices_by_dim.push_back(distance_matrix.size(0));

        this->filtration_values_by_dim.push_back(
            torch::empty({distance_matrix.size(0)}, 
            this->tensopt_real)
            .fill_(0)
            ); 
}


void VietorisRipsArgsGenerator::make_boundary_info_edges(){
    Tensor ba_dim_1, filt_val_vec_dim_1; 
    auto n_edges = binom_coeff_cpu(distance_matrix.size(0), 2); 

    ba_dim_1 = torch::empty({n_edges, 2}, this->tensopt_int); 

    write_combinations_table_to_tensor(ba_dim_1, 0, 0, distance_matrix.size(0)/*=max_n*/, 2/*=r*/);

    // building the vector containing the filtraiton values of the edges 
    // in the same order as they appear in ba_dim_1...
    auto x_indices = ba_dim_1.slice(1, 0, 1).squeeze(); 
    auto y_indices = ba_dim_1.slice(1, 1, 2); 

    // filling filtration vector with edge filtration values ... 
    filt_val_vec_dim_1 = this->distance_matrix.index_select(0, x_indices);
    filt_val_vec_dim_1 = filt_val_vec_dim_1.gather(1, y_indices);
    filt_val_vec_dim_1 = filt_val_vec_dim_1.squeeze(); // 

    // reduce to edges with filtration value <= max_ball_diameter...
    if (max_ball_diameter > 0){
        auto i_select = filt_val_vec_dim_1.le(this->max_ball_diameter).nonzero().squeeze(); //TODO check
        if (i_select.numel() ==  0){

            ba_dim_1 = torch::empty({0}, ba_dim_1.options()); 

            filt_val_vec_dim_1 = torch::empty({0}, filt_val_vec_dim_1.options()); 
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


void VietorisRipsArgsGenerator::make_boundary_info_non_edges(){
    Tensor filt_vals_prev_dim;
    int64_t n_dim_min_one_simplices; 
    Tensor new_boundary_info, new_filt_vals;

    for (int dim = 2; dim <= this->max_dimension; dim++){

        filt_vals_prev_dim = this->filtration_values_by_dim.at(dim - 1);
        n_dim_min_one_simplices = filt_vals_prev_dim.size(0); 

        if (n_dim_min_one_simplices < dim + 1){
            // There are not enough dim-1 simplices ...
            new_boundary_info = torch::empty({0, dim + 1}, this->tensopt_int);

            new_filt_vals = torch::empty({0}, this->tensopt_real);
        }
        else{
            // There are enough dim - 1 simplices ...
            auto combinations = torch::empty(
                {binom_coeff_cpu(n_dim_min_one_simplices, dim + 1), dim + 1}, 
                this->tensopt_int); 

            // write combinations ... 
            write_combinations_table_to_tensor(combinations, 0, 0, n_dim_min_one_simplices, dim + 1); 

            new_boundary_info = co_faces_from_combinations(combinations, this->boundary_info_non_vertices.at(dim - 2)); 

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


void VietorisRipsArgsGenerator::make_simplex_ids_compatible_within_dimensions(){
    
    auto index_offset = this->n_simplices_by_dim.at(0);
    int dim; 
    for (int i=1; i < this->boundary_info_non_vertices.size(); i++){

        dim = i + 1;
        auto boundary_info = this->boundary_info_non_vertices.at(i); 
        boundary_info.add_(index_offset); 
        
        index_offset += this->n_simplices_by_dim.at(dim-1);
    }
}


void VietorisRipsArgsGenerator::make_simplex_dimension_vector(){
    int64_t n_simplices = 0;
    for (int i = 0; i < this->n_simplices_by_dim.size(); i++){
        n_simplices += this->n_simplices_by_dim.at(i); 
    }

    simplex_dimension_vector = torch::empty({n_simplices}, 
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


void VietorisRipsArgsGenerator::make_filtration_values_vector_without_vertices(){
    
    std::vector<Tensor> filt_values_non_vertex_simplices; 
    for (int i = 1; i < this->filtration_values_by_dim.size(); i++){
    
        auto filt_vals = this->filtration_values_by_dim.at(i); 
        filt_values_non_vertex_simplices.push_back(filt_vals);  
    } 

    this->filtration_values_vector_without_vertices = cat(filt_values_non_vertex_simplices, 0);     
}


void VietorisRipsArgsGenerator::do_filtration_add_eps_hack(){
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

        auto filt_add_hack_values = torch::empty(
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


void VietorisRipsArgsGenerator::make_sorting_infrastructure(){
    auto sort_ret = this->filtration_values_vector_without_vertices.sort(0);     
    this->sort_indices_without_vertices = std::get<1>(sort_ret);
    this->sort_indices_without_vertices_inverse = 
        std::get<1>(this->sort_indices_without_vertices.sort(0));
}


void VietorisRipsArgsGenerator::undo_filtration_add_eps_hack(){
    if (this->max_dimension >= 2 && this->n_simplices_by_dim.at(2) > 0){
        this->filtration_values_vector_without_vertices -=
            this->filtration_add_eps_hack_values; 
    }
}


void VietorisRipsArgsGenerator::make_sorted_filtration_values_vector(){

    auto dim_0_filt_values = torch::empty({n_simplices_by_dim.at(0)}, this->tensopt_real);
    
    dim_0_filt_values.fill_(0);

    auto tmp = this->filtration_values_vector_without_vertices
        .index_select(0, this->sort_indices_without_vertices);

    tmp = cat({dim_0_filt_values, tmp}); 

    this->sorted_filtration_values_vector = tmp;  
}


void VietorisRipsArgsGenerator::make_boundary_array_rows_unsorted(){
    auto n_non_vertex_simplices = 0;
    for (int i=1; i < this->n_simplices_by_dim.size(); i++){
        n_non_vertex_simplices += this->n_simplices_by_dim.at(i); 
    }

    auto ba = torch::empty(
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
            torch::empty({n_simplices_by_dim.at(0)}, look_up_table_row.options())
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


void VietorisRipsArgsGenerator::apply_sorting_to_rows(){
    this->boundary_array = boundary_array.index_select(
        0,
        this->sort_indices_without_vertices
    );

    auto simp_dim_slice = this->simplex_dimension_vector.slice(0, this->n_simplices_by_dim.at(0));
    
    simp_dim_slice.copy_(simp_dim_slice.index_select(0, this->sort_indices_without_vertices)); 
}


void VietorisRipsArgsGenerator::make_ba_row_i_to_bm_col_i_vector(){
    auto tmp = torch::empty({this->boundary_array.size(0)}, this->tensopt_int);
    TensorUtils::fill_range_cuda_(tmp); 
    tmp += this->n_simplices_by_dim.at(0); 

    this->ba_row_i_to_bm_col_i_vector = tmp; 
}

std::vector<Tensor> VietorisRipsArgsGenerator::operator()(
    const Tensor & distance_matrix, 
    int64_t max_dimension, 
    double max_ball_diameter){
    
    std::vector<Tensor> ret; 

    this->init_state(distance_matrix, max_dimension, max_ball_diameter); 
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
        ret.push_back(torch::empty({0, 2*(max_dimension + 1)}, this->tensopt_int));
        ret.push_back(torch::empty({0}, this->tensopt_int));
        ret.push_back(torch::zeros({distance_matrix.size(0)}, this->tensopt_int));

        // We generate the 0-vector in this way to ensure that distance_matrix
        // will have zero gradients instead of None after a backward call
        // in pytorch ... 
        auto filtration_values = distance_matrix.slice(1, 0, 1).squeeze().clone();
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
                barcodes = torch::empty({0, 2}, filtration_values.options()); 
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
                    barcodes = torch::empty({0, 2}, filtration_values.options()); 
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
                barcodes = torch::empty({0}, filtration_values.options());
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
    const Tensor& distance_matrix,
    int64_t max_dimension, 
    double max_ball_diameter){    
    
    std::vector<std::vector<Tensor>> ret; 
    auto args_generator = VietorisRipsArgsGenerator();

    auto args = args_generator(distance_matrix, max_dimension, max_ball_diameter);
    
    auto pers = CalcPersCuda::calculate_persistence(
        args.at(0), args.at(1), args.at(2), max_dimension, -1
    );

    auto filtration_values = args.at(3); 
    ret = calculate_persistence_output_to_barcode_tensors(pers, filtration_values); 

    return ret;
}


std::vector<std::vector<Tensor>> vr_persistence_l1(
    const Tensor& point_cloud,
    int64_t max_dimension, 
    double max_ball_diameter){

    auto distance_matrix = l1_norm_distance_matrix(point_cloud);
    
    return vr_persistence(distance_matrix, max_dimension, max_ball_diameter);
}



} // namespace VRCompCuda 