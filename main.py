import logging
import matplotlib.pyplot as plt

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

print("TEST 1: FP16 Configuration")
# Confirming the issues with fp16
config_fp16 = config_mgr.create_config(
    precision=Precision.FP16,
    gpu_type=GPUType.T4,
    runtime=Runtime.TORCH_NATIVE
)


# Change this for the correct configuration when running some tests
pipeline_fp16 = create_pipeline(config_fp16)
image_fp16 = pipeline_fp16.generate(TEST_PROMPT, seed=TEST_SEED)
image_fp16.save("output_fp16.png")

print(f"\n{pipeline_fp16.metrics}")

plt.figure(figsize=(10,10))
plt.imshow(image_fp16)
plt.axis('off')
plt.title("FP16 Output")
plt.show()

pipeline_fp16.cleanup()