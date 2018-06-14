#include <torch/torch.h>

using namespace at;


// CUDA forward declarations

Tensor find_merge_pairings_cuda(Tensor, int);
Tensor merge_columns_cuda(Tensor, Tensor);
Tensor read_points_cuda(Tensor);
Tensor calculate_persistence_cuda(Tensor, Tensor, int);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


Tensor find_merge_pairings(
    Tensor pivots,
    int max_pairs){
  CHECK_INPUT(pivots);
  assert(pivots.type().scalarType() == ScalarType::Int);

  return find_merge_pairings_cuda(pivots, max_pairs);
}


void merge_columns(
    Tensor descending_sorted_boundary_array,
    Tensor merge_pairs);

Tensor read_points(
    Tensor reduced_descending_sorted_boundary_array);

Tensor calculate_persistence(
    Tensor descending_sorted_boundary_array,
    Tensor column_dimension,
    int max_pairs)
{
  CHECK_INPUT(descending_sorted_boundary_array);
  assert(descending_sorted_boundary_array.type().scalarType() == ScalarType::Int);

  auto dgms = calculate_persistence_cuda(descending_sorted_boundary_array,
                                         column_dimension,
                                         max_pairs);
  return dgms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("_find_merge_pairings", &find_merge_pairings, "find_merge_pairings (CUDA)");
  // m.def("_merge_columns", &merge_columns, "merge_columns (CUDA)");
  // m.def("_read_points", &read_points, "read_points (CUDA)");
  m.def("calculate_persistence", &calculate_persistence, "calculate_persistence (CUDA)");
}