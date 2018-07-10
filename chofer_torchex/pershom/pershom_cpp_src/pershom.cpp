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
std::vector<std::vector<Tensor>> calculate_persistence_cuda(Tensor, Tensor, Tensor, int, int);


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
  Tensor compr_desc_sort_ba,
  Tensor merge_pairings, 
  int* h_boundary_array_needs_resize
)
{
  CHECK_INPUT(compr_desc_sort_ba);
  assert(compr_desc_sort_ba.type().scalarType() == ScalarType::Int);

  CHECK_INPUT(merge_pairings);
  assert(merge_pairings.type().scalarType() == ScalarType::Int);

  merge_columns_cuda_kernel_call<int32_t>(compr_desc_sort_ba, merge_pairings, h_boundary_array_needs_resize); 
}


#pragma endregion




// C++ high level interface
#pragma region C++ high level interface

//documentation see ../pershom_backend.py
Tensor find_merge_pairings(
    Tensor pivots,
    int max_pairs = -1){
  CHECK_INPUT(pivots);
  assert(pivots.type().scalarType() == ScalarType::Int);

  return find_merge_pairings_cuda(pivots, max_pairs);
}

//documentation see ../pershom_backend.py
void merge_columns_(
    Tensor compr_desc_sort_ba,
    Tensor merge_pairs){
  CHECK_INPUT(compr_desc_sort_ba);
  assert(compr_desc_sort_ba.type().scalarType() == ScalarType::Int);
  CHECK_INPUT(merge_pairs);
  assert(merge_pairs.type().scalarType() == ScalarType::Long);

  merge_columns_cuda(compr_desc_sort_ba, merge_pairs);

}


//documentation see ../pershom_backend.py
std::vector<std::vector<Tensor>> read_barcodes(
    Tensor pivots,
    Tensor simplex_dimension,
    int max_dimension){

  CHECK_INPUT(pivots);
  assert(pivots.type().scalarType() == ScalarType::Int);
  CHECK_INPUT(simplex_dimension);
  assert(simplex_dimension.type().scalarType() == ScalarType::Int);
  return read_barcodes_cuda(pivots, simplex_dimension, max_dimension);

}

//documentation see ../pershom_backend.py
std::vector<std::vector<Tensor>> calculate_persistence(
    Tensor compr_desc_sort_ba,
    Tensor ind_not_reduced, //TODO rename parameter accordingly to python binding 
    Tensor simplex_dimension,
    int max_dimension,
    int max_pairs){

  CHECK_INPUT(compr_desc_sort_ba);
  assert(compr_desc_sort_ba.type().scalarType() == ScalarType::Int);
  CHECK_INPUT(ind_not_reduced);
  assert(ind_not_reduced.type().scalarType() == ScalarType::Long);
  CHECK_INPUT(simplex_dimension);
  assert(simplex_dimension.type().scalarType() == ScalarType::Int);

  assert(compr_desc_sort_ba.size(0) == ind_not_reduced.size(0));
  assert(ind_not_reduced.ndimension() == 1);
  assert(simplex_dimension.ndimension() == 1);


  auto dgms = calculate_persistence_cuda(compr_desc_sort_ba,
                                         ind_not_reduced, 
                                         simplex_dimension,                                         
                                         max_dimension,
                                         max_pairs);
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

// #include "vr_comp.h"
#include "vr_comp_cuda.cuh"



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

  m.def("vr_persistence", &VRCompCuda::vr_l1_persistence, "test");
  m.def("VRCompCuda::write_combinations_table_to_tensor", &VRCompCuda::write_combinations_table_to_tensor, "");
}


#endif