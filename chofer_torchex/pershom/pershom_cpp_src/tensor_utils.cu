#include <ATen/ATen.h>


using namespace at;


namespace TensorUtils{

namespace {

template<typename scalar_t>
__global__ void fill_range_kernel(scalar_t* out, int64_t out_numel){
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < out_numel){
        out[index] = index;
    }
}
}

void fill_range_cuda_(Tensor t)
{
  // AT_ASSERT(t.type().is_cuda());

    const int threads_per_block = 256;
    const int blocks = t.numel()/threads_per_block + 1;

    auto scalar_type = t.type().scalarType();
    switch(scalar_type)
    {
        case ScalarType::Int: 
        fill_range_kernel<int32_t><<<blocks, threads_per_block>>>(t.data<int32_t>(), t.numel());
        break;

        case ScalarType::Long: 
        fill_range_kernel<int64_t><<<blocks, threads_per_block>>>(t.data<int64_t>(), t.numel());
        break;
        
        default:
        throw std::invalid_argument("Unrecognized Type!");
    }
}
    
} // namespace TensorUtils