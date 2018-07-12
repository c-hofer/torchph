#ifndef PROFILE //TODO remove this?

#include <torch/torch.h>

#include "calc_pers_cuda.cuh"
#include "vr_comp_cuda.cuh"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("CalcPersCuda__find_merge_pairings", &CalcPersCuda::find_merge_pairings, "find_merge_pairings (CUDA)");
  m.def("CalcPersCuda__merge_columns", &CalcPersCuda::merge_columns, "merge_columns (CUDA)");
  m.def("CalcPersCuda__read_barcodes", &CalcPersCuda::read_barcodes, "read_barcodes (CUDA)");
  m.def("CalcPersCuda__calculate_persistence", &CalcPersCuda::calculate_persistence, "calculate_persistence (CUDA)");
  m.def("CalcPersCuda__my_test_f", &CalcPersCuda::my_test_f, "test function");

  m.def("VRCompCuda__vr_l1_persistence", &VRCompCuda::vr_l1_persistence, "");
  m.def("VRCompCuda__write_combinations_table_to_tensor", &VRCompCuda::write_combinations_table_to_tensor, ""),
  m.def("VRCompCuda__vr_l1_generate_calculate_persistence_args", &VRCompCuda::vr_l1_generate_calculate_persistence_args, "");
}


#endif