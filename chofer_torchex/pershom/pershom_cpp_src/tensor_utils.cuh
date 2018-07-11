#pragma once

#include <ATen/ATen.h>


using namespace at;


namespace TensorUtils{
    void fill_range_cuda_(Tensor t);
}