#pragma once


#include <torch/extension.h>


namespace TensorUtils{
    void fill_range_cuda_(Tensor t);
}