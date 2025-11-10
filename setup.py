from torch.utils.cpp_extension import load
import os
CUTLASS_PATH = os.environ.get('CUTLASS_PATH', 'home/aman/cutlass/include')
print(CUTLASS_PATH)
CUDA_ARCH = os.environ.get('CUDA_ARCH', '80')

cutlass_linear_gelu = load(
    name="cutlass_linear_gelu_fp16",
    sources=["csrc/activation/binding.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        f"-I{CUTLASS_PATH}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        f"-gencode=arch=compute_{CUDA_ARCH},code=sm_{CUDA_ARCH}",
        "-Xptxas=-v",  # Verbose register usage
    ],
    extra_ldflags=[],
    verbose=True
)