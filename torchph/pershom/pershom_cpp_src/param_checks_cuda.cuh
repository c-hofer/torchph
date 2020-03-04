#pragma once


#include <torch/extension.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TENSOR_CUDA_CONTIGUOUS(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_TENSOR_INT64(x) AT_ASSERTM(x.type().scalarType() == ScalarType::Long, "expected " #x "to be of scalar type int64")

#define CHECK_SMALLER_EQ(x, y) AT_ASSERTM(x <= y, "expected " #x "<=" #y)
#define CHECK_EQUAL(x, y) AT_ASSERTM(x == y, "expected " #x "==" #y)
#define CHECK_GREATER_EQ(x, y) AT_ASSERTM(x >= y, "expected " #x ">=" #y)

#define CHECK_SAME_DEVICE(x, y) AT_ASSERTM(x.device() == y.device(), #x, #y "are not on same device")


#define PRINT(x) std::cout << x << std::endl