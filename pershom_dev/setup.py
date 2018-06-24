from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pershom_backend_C',
    ext_modules=[
        CUDAExtension('pershom_backend_C_cuda', [
            'pershom_cpp_src/pershom_cuda.cu',
            'pershom_cpp_src/pershom.cpp',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })