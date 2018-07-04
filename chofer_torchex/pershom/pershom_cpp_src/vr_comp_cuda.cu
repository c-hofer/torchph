#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <limits>


using namespace at;


namespace VRCompCuda {
    std::vector<std::vector<Tensor>> vr_persistence_checked(Tensor point_cloud){
    std::vector<std::vector<Tensor>> ret;
    std::vector<Tensor> tmp;

    tmp.push_back(point_cloud);
    ret.push_back(tmp);

    return ret;

    }

} // namespace VRCompCUda 