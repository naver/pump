# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# if you want to compile for all possible CUDA architectures
all_cuda_archs = [] #cuda.get_gencode_flags().replace('compute=','arch=').split()

setup(
    name='cuda_deepm',
    ext_modules = [
        CUDAExtension(
                name = 'cuda_deepm',
                sources = ["func.cpp", "kernels.cu"],
                extra_compile_args = dict(nvcc=['-O2']+all_cuda_archs, cxx=['-O2'])
                )
    ],
    cmdclass = {
        'build_ext': BuildExtension
    })

