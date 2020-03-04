#pragma once


#include <torch/extension.h>


using namespace torch;

namespace CalcPersCuda
{

Tensor find_merge_pairings(
    const Tensor & pivots,
    int64_t max_pairs = -1);

void merge_columns(
    const Tensor & compr_desc_sort_ba,
    const Tensor & merge_pairs);

std::vector<std::vector<Tensor>> read_barcodes(
    const Tensor & pivots,
    const Tensor & simplex_dimension,
    int64_t max_dim_to_read_of_reduced_ba);

std::vector<std::vector<Tensor>> calculate_persistence(
    const Tensor &  compr_desc_sort_ba,
    const Tensor & ba_row_i_to_bm_col_i,
    const Tensor & simplex_dimension,
    int64_t max_dim_to_read_of_reduced_ba,
    int64_t max_pairs);

} // namespace CalcPersCuda