#include <torch/torch.h>

using namespace at;

// CUDA forward declarations

//low level
template <typename scalar_t> Tensor find_slicing_indices_cuda_kernel_call(Tensor);
template <typename scalar_t> void merge_columns_cuda_kernel_call(Tensor, Tensor, int*);


//high level 
Tensor find_merge_pairings_cuda(Tensor, int);
Tensor merge_columns_cuda(Tensor, Tensor);
std::vector<std::vector<Tensor>> read_barcodes_cuda(Tensor, Tensor, int);
std::vector<std::vector<Tensor>> calculate_persistence_cuda(Tensor, Tensor, int, int);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


// C++ low level interface
#pragma region C++ low level interface


Tensor find_slicing_indices_kernel_call(
  Tensor t)
{
  CHECK_INPUT(t);
  assert(t.type().scalarType() == ScalarType::Int);

  return find_slicing_indices_cuda_kernel_call<int32_t>(t);
}


void merge_columns_kernel_call(
  Tensor descending_sorted_boundary_array,
  Tensor merge_pairings, 
  int* h_boundary_array_needs_resize
)
{
  CHECK_INPUT(descending_sorted_boundary_array);
  assert(descending_sorted_boundary_array.type().scalarType() == ScalarType::Int);

  CHECK_INPUT(merge_pairings);
  assert(merge_pairings.type().scalarType() == ScalarType::Int);

  merge_columns_cuda_kernel_call<int32_t>(descending_sorted_boundary_array, merge_pairings, h_boundary_array_needs_resize); 
}


#pragma endregion




// C++ high level interface
#pragma region C++ high level interface


Tensor find_merge_pairings(
    Tensor pivots,
    int max_pairs)
{
  CHECK_INPUT(pivots);
  assert(pivots.type().scalarType() == ScalarType::Int);

  return find_merge_pairings_cuda(pivots, max_pairs);
}

void merge_columns_(
    Tensor descending_sorted_boundary_array,
    Tensor merge_pairs)
{
  CHECK_INPUT(descending_sorted_boundary_array);
  assert(descending_sorted_boundary_array.type().scalarType() == ScalarType::Int);
  CHECK_INPUT(merge_pairs);
  assert(merge_pairs.type().scalarType() == ScalarType::Long);

  merge_columns_cuda(descending_sorted_boundary_array, merge_pairs);
}

std::vector<std::vector<Tensor>> read_barcodes(
    Tensor pivots,
    Tensor column_dimension,
    int max_dimension)
{

  CHECK_INPUT(pivots);
  assert(pivots.type().scalarType() == ScalarType::Int);
  CHECK_INPUT(column_dimension);
  assert(column_dimension.type().scalarType() == ScalarType::Int);
  return read_barcodes_cuda(pivots, column_dimension, max_dimension);
}

std::vector<std::vector<Tensor>> calculate_persistence(
    Tensor descending_sorted_boundary_array,
    Tensor column_dimension,
    int max_pairs,
    int max_dimension)
{

  CHECK_INPUT(descending_sorted_boundary_array);
  assert(descending_sorted_boundary_array.type().scalarType() == ScalarType::Int);
  CHECK_INPUT(column_dimension);
  assert(column_dimension.type().scalarType() == ScalarType::Int);

  auto dgms = calculate_persistence_cuda(descending_sorted_boundary_array,
                                         column_dimension,
                                         max_pairs,
                                         max_dimension);
  return dgms;
}


#pragma endregion


//-------------devel
Tensor my_test_f_cuda(Tensor);

Tensor my_test_f(Tensor t)
{
  auto state = globalContext().getTHCState();

  // Context* c = &globalContext();
  // auto thc_state = c->getTHCState();
  
  // std::cout << p_state << std::endl;
  // return my_test_f_cuda(t);
  return t;
}
//-------------


#ifndef PROFILE


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // low level
  m.def("find_slicing_indices_kernel_call", &find_slicing_indices_kernel_call, "find_slicing_indices_kernel_call (CUDA)");
  m.def("merge_columns_kernel_call", &merge_columns_kernel_call, "merge_columns_kernel_call (CUDA)");

  //high level 
  m.def("find_merge_pairings", &find_merge_pairings, "find_merge_pairings (CUDA)");
  m.def("merge_columns_", &merge_columns_, "merge_columns (CUDA)");
  m.def("read_barcodes", &read_barcodes, "read_barcodes (CUDA)");
  m.def("calculate_persistence", &calculate_persistence, "calculate_persistence (CUDA)");
  m.def("my_test_f", &my_test_f, "test function");
}


#endif