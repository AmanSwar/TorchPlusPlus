from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob


CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "~/cutlass/include")
CUTLASS_PATH = os.path.abspath(os.path.expanduser(CUTLASS_PATH))

CUDA_ARCH = os.environ.get("CUDA_ARCH", "80")
gencode_flag = f"-gencode=arch=compute_{CUDA_ARCH},code=sm_{CUDA_ARCH}"

# Base compilation arguments shared by all extensions
BASE_NVCC_ARGS = [
    "-O2",
    "-std=c++17",
    f"-I{CUTLASS_PATH}",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    gencode_flag,
    "-Xptxas=-v",
    "-Xcompiler",
    "-fPIC",
]

BASE_CXX_ARGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
]

# Include paths for all CUDA files
INCLUDE_DIRS = [
    "csrc",
    CUTLASS_PATH,
]


# Define all CUDA extensions
def get_extensions():
    extensions = [
        CUDAExtension(
            name="linearActvationFp16",
            sources=["csrc/activation/binding.cu"],
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                "cxx": BASE_CXX_ARGS,
                "nvcc": BASE_NVCC_ARGS,
            },
        ),
        CUDAExtension(
            name="ropeApplyFunction",
            sources=["csrc/dlops/rope/rope_binding.cu"],
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                "cxx": BASE_CXX_ARGS,
                "nvcc": BASE_NVCC_ARGS,
            },
        ),
        CUDAExtension(
            name="rmsnormFused",
            sources=["csrc/dlops/rmsnorm/rmsnorm_binding.cu"],
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                "cxx": BASE_CXX_ARGS,
                "nvcc": BASE_NVCC_ARGS,
            },
        ),
        CUDAExtension(
            name="layernormFused",
            sources=["csrc/dlops/layernorm/layernorm_binding.cu"],
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                "cxx": BASE_CXX_ARGS,
                "nvcc": BASE_NVCC_ARGS,
            },
        ),
    ]
    return extensions


setup(
    name="torchpp",
    version="0.1.0",
    description="PyTorch Performance Plus - High-performance CUDA kernels for deep learning",
    author="Aman",
    packages=find_packages(exclude=["tests", "examples", "refs", "build"]),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
