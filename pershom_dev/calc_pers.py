import torch
import chofer_torchex.pershom.pershom_backend as pershom_backend

device = torch.device('cuda')
dtype = torch.int64

ba = torch.empty([0, 2], device=device, dtype=dtype)
ba_row_i_to_bm_col_i = ba
simplex_dimension = torch.zeros(10, device=device, dtype=dtype)
print(simplex_dimension)

# ret = pershom_backend.__C.CalcPersCuda__calculate_persistence(ba, ba_row_i_to_bm_col_i, simplex_dimension, 3, -1)
ret = pershom_backend.__C.CalcPersCuda__my_test_f(ba); 

print(ret)