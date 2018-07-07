#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

#include"param_checks_cuda.cuh"

using namespace at;


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
    int r = blockIdx.x*blockDim.x + threadIdx.x;
    int c = blockIdx.y*blockDim.y + threadIdx.y;

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
Tensor binomial_table(int64_t max_n, int64_t max_k, const Type& type){
 
    
    auto ret = type.toScalarType(ScalarType::Long).tensor({max_k, max_n}); //LBL: creation 

    dim3 threads_per_block = dim3(64, 64);
    dim3 num_blocks= dim3(max_k/threads_per_block.y + 1, max_n/threads_per_block.x + 1);    


    binomial_table_kernel<<<threads_per_block, num_blocks>>>(
        ret.data<int64_t>(),
        max_n, 
        max_k);        

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

    for (int i=0; i<r;i++){
        bt_row = &binom_table[(r - i - 1)*max_n];

        for (int j=0; j < max_n - 1; j++){
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
    for (int i = 0; i < r; i++){        
        if (out[r - i - 2] > out[r - i - 1] + 1){
            out[r - i - 1] += 1;
           
            // fill the following digits with the smallest ordered sequence ... 
            for (int j=0; j < i; j++){
                out[r - j - 1] = j;
            }
            return;
        }
    }

    // If the first digit has to be increased ...
    out[0] += 1;    
    // fill the following digits with the smallest ordered sequence ... 
    for (int j=0; j < r - 1; j++){
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

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

    if (thread_id*n_comb_by_thread < binom_coeff(max_n, r)){
        int64_t* comb = new int64_t[r]; 
        unrank_combination(comb, thread_id*n_comb_by_thread, max_n, r, binom_table);

        for (int i=0; i < n_comb_by_thread; i++){

            if (thread_id*n_comb_by_thread + i >= n_max_over_r) break; 

            for (int j=0; j < r; j++){
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
    CHECK_SMALLER_EQ(out.size(1), r);  
    CHECK_EQUAL(out.ndimension(), 2);


    const auto bt = binomial_table(max_n, r, out.type());
    const int64_t n_comb_by_thread = 100; //TODO optimize
    const int64_t threads_per_block = 64; //TODO optimize

    const int64_t blocks = n_max_over_r/(threads_per_block*n_comb_by_thread) + 1; 


    write_combinations_to_tensor_kernel<<<threads_per_block, blocks>>>(
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
}

#pragma endregion 







std::vector<std::vector<Tensor>> vr_l1_persistence_checked(
    const Tensor& point_cloud,
    double max_ball_radius){


    CHECK_INPUT(point_cloud);
    std::vector<std::vector<Tensor>> ret;
    std::vector<Tensor> tmp;

    auto max_n = 10; 
    auto r = 3; 

    auto bt = binomial_table(100, 3, point_cloud.type().toScalarType(ScalarType::Long));
    tmp.push_back(bt);

    auto tensor = point_cloud.type().toScalarType(ScalarType::Long).tensor({124, 3}).fill_(-1);
    write_combinations_table_to_tensor(tensor, 2, 100, max_n, r); 
    // auto tensor = binom_n_over_2_desc_sort_cuda(1000, point_cloud.type());
    // auto tensor = point_cloud.type();

    // auto tensor_gpu = tensor.toBackend(Backend::CUDA);

    tmp.push_back(tensor);
    ret.push_back(tmp);

    return ret;
}

} // namespace VRCompCuda 