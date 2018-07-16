#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>

#include "param_checks_cuda.cuh"
#include "tensor_utils.cuh"
#include "calc_pers_cuda.cuh"

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
Tensor binomial_table(int64_t max_n, int64_t max_k, const Type& type){
 
    
    auto ret = type.toScalarType(ScalarType::Long).tensor({max_k, max_n}); //LBL: creation 

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


    const auto bt = binomial_table(max_n, r, out.type());
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


std::tuple<Tensor, Tensor> get_boundary_and_filtration_info_dim_1(
    const Tensor & point_cloud, 
    double max_ball_radius){

    Tensor ba_dim_1, filt_val_vec_dim_1; 
    auto n_edges = binom_coeff_cpu(point_cloud.size(0), 2); 
    ba_dim_1 = point_cloud.type().toScalarType(ScalarType::Long).tensor({n_edges, 2}); 

    write_combinations_table_to_tensor(ba_dim_1, 0, 0, point_cloud.size(0)/*=max_n*/, 2/*=r*/);

    auto distance_matrix = l1_norm_distance_matrix(point_cloud); 

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
            ba_dim_1 = ba_dim_1.type().tensor({0});
            filt_val_vec_dim_1 = filt_val_vec_dim_1.type().tensor({0}); 
        }
        else{
            ba_dim_1 = ba_dim_1.index_select(0, i_select);
            filt_val_vec_dim_1 = filt_val_vec_dim_1.index_select(0, i_select); 
        }
    }

    return std::make_tuple(ba_dim_1, filt_val_vec_dim_1);
}


std::tuple<Tensor, Tensor> get_boundary_and_filtration_info(
    const Tensor & filt_vals_prev_dim, 
    int64_t dim){

    auto n_dim_min_one_simplices = filt_vals_prev_dim.size(0); 

    Tensor new_boundary_info, new_filt_vals;

    if (n_dim_min_one_simplices < dim + 1){
        // There are not enough dim-1 simplices ...
        new_boundary_info = filt_vals_prev_dim.type().toScalarType(ScalarType::Long).tensor({0, dim + 1});
        new_filt_vals = filt_vals_prev_dim.type().tensor({0});
    }
    else{
        // There are enough dim-1 simplices ...
        auto n_new_simplices = binom_coeff_cpu(n_dim_min_one_simplices, dim + 1); 
        auto n_simplices_prev_dim = filt_vals_prev_dim.size(0); 

        new_boundary_info = filt_vals_prev_dim.type().toScalarType(ScalarType::Long).tensor({n_new_simplices, dim + 1}); 

        // write combinations ... 
        write_combinations_table_to_tensor(new_boundary_info, 0, 0, n_simplices_prev_dim, dim + 1); 
        cudaStreamSynchronize(0); 

        auto bi_cloned = new_boundary_info.clone(); // we have to clone here other wise auto-grad does not work!
        new_filt_vals = filt_vals_prev_dim.expand({n_new_simplices, filt_vals_prev_dim.size(0)});
        new_filt_vals = new_filt_vals.gather(1, bi_cloned); 
        new_filt_vals = std::get<0>(new_filt_vals.max(1));

        // If we have just one simplex of the current dimension this
        // condition avoids that new_filt_vals is squeezed to a 0-dim 
        // Tensor
        if (new_filt_vals.ndimension() != 1){      
            new_filt_vals = new_filt_vals.squeeze(); 
        }
    }

    return std::make_tuple(new_boundary_info, new_filt_vals); 
}


//TODO refactor 
std::vector<Tensor> vr_l1_generate_calculate_persistence_args(
    const Tensor& point_cloud,
    int64_t max_dimension, 
    double max_ball_radius
    ){

    CHECK_TENSOR_CUDA_CONTIGUOUS(point_cloud);
    CHECK_SMALLER_EQ(max_dimension + 1, point_cloud.size(0)); 
    CHECK_SMALLER_EQ(0, max_ball_radius);


    std::vector<Tensor> ret;
    Type& Long = point_cloud.type().toScalarType(ScalarType::Long);

    // 1. generate boundaries and filtration values ...

    // boundary_and_filtration_info_by_dim[i] == (enumerated boundary combinations, filtration values) of 
    // dimension i + 1. 
    std::vector<std::tuple<Tensor, Tensor>> boundary_and_filtration_by_dim;

    boundary_and_filtration_by_dim.push_back(
        get_boundary_and_filtration_info_dim_1(point_cloud, max_ball_radius)
    );

    for (int dim = 2; dim <= max_dimension; dim++){
        auto filt_vals_prev_dim = std::get<1>(boundary_and_filtration_by_dim.at(dim - 1 - 1));

        boundary_and_filtration_by_dim.push_back(
            get_boundary_and_filtration_info(filt_vals_prev_dim, dim)
        );
    }

    // 2. Create helper structure which contains meta info about simplex numbers ... 
    int64_t n_non_vertex_simplices = 0;
    int64_t n_simplices = point_cloud.size(0); 
    std::vector<int64_t> n_simplices_by_dim; 
    n_simplices_by_dim.push_back(point_cloud.size(0)); 

    for (int i = 0; i < boundary_and_filtration_by_dim.size(); i++){
        auto boundary_info = std::get<0>(boundary_and_filtration_by_dim.at(i));
        n_non_vertex_simplices += boundary_info.size(0); 
        n_simplices += boundary_info.size(0); 
        n_simplices_by_dim.push_back(boundary_info.size(0)); 
    }

    // TODO returning in mid of function is not nice. Can we improve this? 
    // If there are only vertices, we return the empty vector 
    // and let the caller handle the problem ... 
    if (n_non_vertex_simplices == 0){
        return ret; 
    }

    // 3. Make simplex id's compatible within dimensions ... 
    /*    
    In order to keep indices in the boundary info tensors 
    compatible within dimensions we have to add an offset
    to the enumerated combinations, starting with 
    dimension 2 simplices (the boundaries of dim 1 simplices are vertices, 
    hence the enumeration of the boundary combinations is valid)
    */
    auto index_offset = n_simplices_by_dim.at(0);
    for (int i=1; i < boundary_and_filtration_by_dim.size(); i++){
        auto boundary_info = std::get<0>(boundary_and_filtration_by_dim.at(i)); 
        boundary_info.add_(index_offset); 

        auto n_simplices_in_prev_dim = std::get<0>(boundary_and_filtration_by_dim.at(i-1)).size(0); 
        index_offset += n_simplices_in_prev_dim;
    }    


    // 4. Create simplex_dimension vector ... 
    auto simplex_dimension = Long.tensor(n_non_vertex_simplices + n_simplices_by_dim.at(0)); 

    {
        int64_t copy_offset = 0; 
        for (int i = 0; i <= max_dimension; i++){
            simplex_dimension.slice(0, copy_offset, copy_offset + n_simplices_by_dim.at(i)).fill_(i); 
            copy_offset += n_simplices_by_dim.at(i); 
        }
    }


    // 5. Create filtration vector ... 
    Tensor filtration_values_vector;
    {
        std::vector<Tensor> filt_values_non_vertex_simplices; 
        for (int i = 0; i < boundary_and_filtration_by_dim.size(); i++){
        
            auto filt_vals = std::get<1>(boundary_and_filtration_by_dim.at(i));
            filt_values_non_vertex_simplices.push_back(filt_vals);  
        } 

        filtration_values_vector = cat(filt_values_non_vertex_simplices, 0); 
    }    

    // This is a dirty hack to ensure that simplices do not occour before their boundaries 
    // in the filtration. As the filtration is raised to higher dimensional simplices by 
    // taking the maxium of the involved edge filtration values and sorting does not guarantee
    // a specific ordering in case of equal values we are forced to ensure a well defined 
    // filtration by adding an increasing epsilon to each dimension. Later this has to be 
    // substracted again. 
    // Example: f([1,2,3]) = max(f([1,2]), f([3,1]), f([2,3])) --> w.l.o.g. f([1,2,3]) == f([1,2])
    // Hence we set f([1,2,3]) = f([1,2]) + epsilon
    auto filt_add_hack_values = filtration_values_vector.type().tensor({filtration_values_vector.size(0)}).fill_(0);
    
    {
        if (max_dimension >= 2 && n_simplices_by_dim.at(2) > 0){
            
            // we take epsilon of float to ensure that it is well defined even if 
            // we decide to alter the floating point type of the filtration values 
            // realm 
            float add_const_base_value = 100 * std::numeric_limits<float>::epsilon(); // multily with 100 to be save against rounding issues
            auto copy_offset = n_simplices_by_dim.at(1); 

            for (int dim = 2; dim <= max_dimension; dim++){
                filt_add_hack_values.slice(0, copy_offset, copy_offset + n_simplices_by_dim.at(dim))
                    .fill_(add_const_base_value); 

                add_const_base_value += add_const_base_value; 
                copy_offset += n_simplices_by_dim.at(dim); 
            }

            filtration_values_vector += filt_add_hack_values;
        }

        filt_add_hack_values = filt_add_hack_values.clone();
    
    }

    //6 Do sorting ...
    
    auto sort_filt_res = filtration_values_vector.sort(0);
    auto sorted_filtration_values_vector = std::get<0>(sort_filt_res);
    auto sort_i_filt = std::get<1>(sort_filt_res); 

    // revert filtration hack if necessary ...
    if (max_dimension >= 2 && n_simplices_by_dim.at(2) > 0){
        filt_add_hack_values = filt_add_hack_values.index_select(0, sort_i_filt); 
        sorted_filtration_values_vector -= filt_add_hack_values;
    }
    // now the filtration is cleaned and we can continue. 

    // Simplex ids in boundary_array entries include vertices.
    // As filtration_value_vector so far starts with edges we have to take care of this. 
    auto dim_0_filt_values = sorted_filtration_values_vector.type().zeros({n_simplices_by_dim.at(0)}); 
    sorted_filtration_values_vector = cat({dim_0_filt_values, sorted_filtration_values_vector}, 0); 
  

    // Sort simplex_dimension ...
    simplex_dimension.slice(0, n_simplices_by_dim.at(0)) = 
        simplex_dimension.slice(0, n_simplices_by_dim.at(0)).index_select(0, sort_i_filt);

    // Copy boundary_info of each dimension into the final boundary array ... 
    auto boundary_array = point_cloud.type().toScalarType(ScalarType::Long)
        .tensor({n_non_vertex_simplices, 2*(max_dimension + 1)});

    {
        boundary_array.fill_(-1); 

        // copy edges ... 
        auto edge_boundary_info = std::get<0>(boundary_and_filtration_by_dim.at(0));
        boundary_array.slice(0, 0, n_simplices_by_dim.at(1)).slice(1, 0, 2) = edge_boundary_info; 

        // copy higher dimensional simplices
        if (max_dimension >= 2){
            // we need a look up table which lets us change the simplex ids we get from the initial 
            // enumeration (write_combinations_table_to_tensor) to the id the have w.r.t. the ordering
            // of the filtration values. We create this table now ...

            // This gives us the mapping id -> new_id w.r.t. sorting by filtration values ...
            auto look_up_table_row = std::get<1>(sort_i_filt.sort(0));

            // look_up_table_row is yet based on id's without vertices, we adapt theis now ...
            auto dummy_sort_indices = sort_i_filt.type().tensor({n_simplices_by_dim.at(0)}).fill_(std::numeric_limits<int64_t>::max());
            look_up_table_row = look_up_table_row + n_simplices_by_dim.at(0); 

            // as vertices have no boundary we will never select a value of the first #vertices entries, 
            // but we need look_up_table_row.size(0) == #simplices in order to get a consistent mapping...
            look_up_table_row = cat({dummy_sort_indices, look_up_table_row}, 0); 

            int64_t copy_offset = n_simplices_by_dim.at(1);             

            for (int i = 1; i < max_dimension; i++){

                auto boundary_info = std::get<0>(boundary_and_filtration_by_dim.at(i)); 

                if (boundary_info.size(0) == 0){
                    continue; 
                }

                auto look_up_table = look_up_table_row.expand({boundary_info.size(0), look_up_table_row.size(0)});  

                // Apply ordering to row content ... 
                boundary_info = look_up_table.gather(1, boundary_info); 

                // Apply ordering to rows ...
                boundary_info = std::get<0>(boundary_info.sort(1, /*descending=*/true));

                boundary_array.slice(0, copy_offset, copy_offset + boundary_info.size(0)).slice(1, 0, boundary_info.size(1))
                     = boundary_info; 

                copy_offset += boundary_info.size(0); 
            }
        }
    }

    // Sort boundary_array rows ...
    boundary_array = boundary_array.index_select(0, sort_i_filt);  

    //7. generate ba_row_i_to_bm_col_i
    auto ba_row_i_to_bm_col_i = boundary_array.type().tensor({boundary_array.size(0)});
    TensorUtils::fill_range_cuda_(ba_row_i_to_bm_col_i); 
    ba_row_i_to_bm_col_i += n_simplices_by_dim.at(0); 

    //8. returning ... 
    ret.push_back(boundary_array); 
    ret.push_back(ba_row_i_to_bm_col_i);
    ret.push_back(simplex_dimension); 
    ret.push_back(sorted_filtration_values_vector);  

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
                barcodes = filtration_values.type().tensor({0, 2}); 
            }
            else {
                birth_i = birth_death_i.slice(1, 0, 1).squeeze(); 
                births = filtration_values.index_select(0, birth_i);

                death_i = birth_death_i.slice(1, 1, 2).squeeze();
                deaths = filtration_values.index_select(0, death_i);

                i_birth_ne_death = births.ne(deaths).nonzero().squeeze(); 
                births = births.index_select(0, i_birth_ne_death);
                deaths = deaths.index_select(0, i_birth_ne_death);

                barcodes = stack({births, deaths}, 1); 

                
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
                barcodes = filtration_values.type().tensor({0, 1});
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


std::vector<std::vector<Tensor>> vr_l1_persistence(
    const Tensor& point_cloud,
    int64_t max_dimension, 
    double max_ball_radius){

    

    auto tmp = vr_l1_generate_calculate_persistence_args(
        point_cloud, max_dimension, max_ball_radius
    );

    auto pers = CalcPersCuda::calculate_persistence(
        tmp.at(0), tmp.at(1), tmp.at(2), max_dimension, -1
    );

    auto filtration_values = tmp.at(3); 
    auto ret = calculate_persistence_output_to_barcode_tensors(pers, filtration_values); 

    return ret;
}


} // namespace VRCompCuda 