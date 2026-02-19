import logging
import time
import torch
import torchvision.transforms as T

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import PRXTransformer2DModel
from transformers import AutoTokenizer
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder

from dataclasses import dataclass

@dataclass
class InferenceMetrics:
    # Used to follow some common inference metrics
    model_load_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    config_id: str = ""

    def __str__(self):
        return (
            f"Config: {self.config_id}\n"
            f"  Load: {self.model_load_time_ms:.0f}ms\n"
            f"  Inference: {self.inference_time_ms:.0f}ms\n"
            f"  Total: {self.total_time_ms:.0f}ms\n"
            f"  Memory: {self.peak_memory_mb:.0f}MB"
        )

class PRXPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = config.precision.torch_dtype

        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.scheduler = None

        self._initialized = False
        self._metrics = InferenceMetrics(config_id=config.config_id)

    def initialize(self):
        if self._initialized:
            return

        start = time.time()
        self.logger.info(f"Loading models: {self.config.config_id}")

        # Load all components
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.checkpoint, subfolder="tokenizer", trust_remote_code=True
        )

        self.text_encoder = T5GemmaEncoder.from_pretrained(
            self.config.checkpoint,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        self.vae = AutoencoderKL.from_pretrained(
            self.config.checkpoint,
            subfolder="vae",
            torch_dtype=self.dtype
        ).to(self.device)

        self.transformer = PRXTransformer2DModel.from_pretrained(
            self.config.checkpoint,
            subfolder="transformer",
            torch_dtype=self.dtype
        ).to(self.device)

        # Apply torch.compile if needed.
        if self.config.runtime == Runtime.TORCH_COMPILE:
            self.logger.info("Compiling transformer...")
            self.transformer = torch.compile(
                self.transformer, mode="reduce-overhead", fullgraph=False
            )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.checkpoint, subfolder="scheduler"
        )

        self._metrics.model_load_time_ms = (time.time() - start) * 1000
        self._initialized = True
        self.logger.info(f"Loaded in {self._metrics.model_load_time_ms:.0f}ms")

    def generate(self, prompt, resolution_width=512, resolution_height=512,
                 num_inference_steps=28, seed=None):
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        if seed is not None:
            torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.time()

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True,
        )

        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].bool().to(self.device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            # Text embeddings
            embeddings = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )['last_hidden_state']

            # Prepare latents
            latent_channels = int(self.vae.config.latent_channels)
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            latent_h = resolution_height // vae_scale_factor
            latent_w = resolution_width // vae_scale_factor

            shape = (1, latent_channels, latent_h, latent_w)
            latents = torch.randn(shape, device=self.device, dtype=self.dtype)

            # Denoising
            for t in self.scheduler.timesteps:
                timestep = t.expand(latents.shape[0])
                noise_pred = self.transformer(
                    latents,
                    timestep / self.scheduler.config.num_train_timesteps,
                    encoder_hidden_states=embeddings,
                    attention_mask=attention_mask,
                ).sample

                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

            # Decode
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor
            ).sample + self.vae.config.shift_factor

            image = (image / 2 + 0.5).clamp(0, 1).to(torch.float32)

        image = T.ToPILImage()(image[0].cpu())

        self._metrics.inference_time_ms = (time.time() - start) * 1000
        if torch.cuda.is_available():
            self._metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

        return image

    @property
    def metrics(self):
        self._metrics.total_time_ms = (
            self._metrics.model_load_time_ms + self._metrics.inference_time_ms
        )
        return self._metrics

    def cleanup(self):
        # freeing some memory after use
        del self.text_encoder, self.vae, self.transformer
        self.text_encoder = self.vae = self.transformer = None
        self._initialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_pipeline(config):
    pipeline = PRXPipeline(config)
    pipeline.initialize()
    return pipeline

print("Loaded pipeline implementation")