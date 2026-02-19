import torch
from enum import Enum

class Precision(Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"

    @property
    def torch_dtype(self):
        return {
            Precision.FP16: torch.float16,
            Precision.BF16: torch.bfloat16,
            Precision.FP32: torch.float32,
        }[self]

class GPUType(Enum):
    T4 = "t4"
    H100 = "h100"

    # Added some properties of the two benchmarked GPUs
    # The compute capability I am using, the memory at the moment no
    @property
    def compute_capability(self):
        return {GPUType.T4: (7, 5), GPUType.H100: (9, 0)}[self]

    @property
    def memory_gb(self):
        return {GPUType.T4: 16, GPUType.H100: 80}[self]

class Runtime(Enum):
    TORCH_NATIVE = "torch_native"
    TORCH_COMPILE = "torch_compile"
