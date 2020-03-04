#pragma once


#include <torch/extension.h>


using namespace torch;


namespace TensorUtils{
    void fill_range_cuda_(Tensor t);
}