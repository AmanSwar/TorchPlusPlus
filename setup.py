from setuptools import setup
from torch.utils.cpp_extension import load , BuildExtension , CUDAExtension
import os


CUTLASS_PATH = os.environ.get('CUTLASS_PATH', '~/cutlass/include')
CUTLASS_PATH = os.path.abspath(os.path.expanduser(CUTLASS_PATH))


CUDA_ARCH = os.environ.get('CUDA_ARCH', '80')

gencode_flag = f"-gencode=arch=compute_{CUDA_ARCH},code=sm_{CUDA_ARCH}"

setup(
    name="LinearActivationFp16",
    ext_modules=[
        CUDAExtension(
            name="linearActvationFp16",
            sources=["csrc/activation/binding.cu"],
            extra_compile_args={
                "cxx": [
                    "-std=c++17",
                    "-O3",
                    "-fPIC",
                ],
                "nvcc": [
                    "-O2",
                    # keep std for host compiler compatibility; nvcc accepts it too
                    "-std=c++17",
                    f"-I{CUTLASS_PATH}",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    gencode_flag,
                    "-Xptxas=-v",
                    "-Xcompiler", "-fPIC",
                ],
            },
        ),
        CUDAExtension(
            name="ropeApplyFunction",   
            sources=["csrc/dlops/rope/rope_binding.cu"],
            extra_compile_args={
                "cxx": [
                    "-std=c++17",
                    "-O3",
                    "-fPIC",
                ],
                "nvcc": [
                    "-O2",
                    # keep std for host compiler compatibility; nvcc accepts it too
                    "-std=c++17",
                    f"-I{CUTLASS_PATH}",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    gencode_flag,
                    "-Xptxas=-v",
                    "-Xcompiler", "-fPIC",
                ],
            },
            
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)