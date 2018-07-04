#pragma once;

#include <ATen/ATen.h>


using namespace at;


namespace VRCompCuda {
    std::vector<std::vector<Tensor>> vr_persistence_checked(Tensor);
}