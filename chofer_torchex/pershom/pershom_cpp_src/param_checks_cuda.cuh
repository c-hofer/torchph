#pragma once

#include <ATen/ATen.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_SMALLER_EQ(x, y) AT_ASSERTM(x <= y, "expected" #x "<=" #y)
#define CHECK_EQUAL(x, y) AT_ASSERTM(x == y, "expected " #x "==" #y)
