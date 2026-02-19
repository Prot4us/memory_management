import logging
import torch
import matplotlib.pyplot as plt

from config_management.enums import GPUType, Precision, Runtime
from config_management.config import ConfigurationManager


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    TEST_PROMPT = (
        "Hundreds of paper lanterns drifting along a quiet river at dusk, "
        "soft orange light piercing cold blue mist, reflections trembling "
        "across rippled water, camera at water level with shallow DOF, "
        "cinematic color contrast of warm and cool tones, shot on Sony Venice 2 "
        "with Cooke S4 50mm lens, f/1.8, ISO 800, graded on Kodak 2383 film LUT"
    )
    TEST_SEED = 42

    config_mgr = ConfigurationManager()

    logging.info("Register and validate the configurations we want to test")

    # Configurations to test:
    # - FP32, FP16, BF16 with torch native
    # - FP32, FP16 with torch.compile
    l_config_fp_32 = {"precision": Precision.FP32, "gpu_type": GPUType.T4, "runtime": Runtime.TORCH_NATIVE}
    l_config_fp_16 = {"precision": Precision.FP16, "gpu_type": GPUType.T4, "runtime": Runtime.TORCH_NATIVE}
    l_config_bf_16 = {"precision": Precision.BF16, "gpu_type": GPUType.T4, "runtime": Runtime.TORCH_NATIVE}
    l_config_fp_32_comp = {"precision": Precision.FP32, "gpu_type": GPUType.T4, "runtime": Runtime.TORCH_COMPILE}
    l_config_fp_16_comp = {"precision": Precision.FP16, "gpu_type": GPUType.T4, "runtime": Runtime.TORCH_COMPILE}
    configurations = [l_config_fp_32, l_config_fp_16, l_config_bf_16, l_config_fp_32_comp, l_config_fp_16_comp]
    for config in configurations:
        config_mgr.register_config(**config)

    # Display current pipelines
    logging.info(config_mgr.list_configs())


    # Select a pipeline and run it


if __name__ == "__main__":
    main()