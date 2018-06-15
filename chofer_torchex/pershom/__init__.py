import os.path as pth

from torch.utils.cpp_extension import load

this_file_dir = pth.dirname(pth.realpath(__file__))
cpp_src_dir = pth.join(this_file_dir, 'pershom_cpp_src')

pershom_cuda_ext = load(
    'pershom_cuda_ext',
    [pth.join(cpp_src_dir, 'pershom.cpp'), 
    pth.join(cpp_src_dir, 'pershom_cuda.cu')], 
    verbose=True)

_find_merge_pairings = pershom_cuda_ext._find_merge_pairings
_read_barcodes = pershom_cuda_ext._read_barcodes
calculate_persistence = pershom_cuda_ext.calculate_persistence
my_test_f = pershom_cuda_ext.my_test_f
