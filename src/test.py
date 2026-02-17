# Helper functions for extracting internal information from the
# various pieces of the model
def test_text_encoder_simple(pipeline, prompt):
    # Extract text encoder output
    tokens = pipeline.tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
        return_attention_mask=True,
    )

    input_ids = tokens['input_ids'].to(pipeline.device)
    attention_mask = tokens['attention_mask'].bool().to(pipeline.device)

    with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.dtype):
        embeddings = pipeline.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )['last_hidden_state']

    return embeddings, input_ids, attention_mask

def test_transformer_simple(pipeline, embeddings, attention_mask, seed=42):
    # Get the latent information from the transformer
    latent_channels = int(pipeline.vae.config.latent_channels)
    vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
    latent_h = 512 // vae_scale_factor
    latent_w = 512 // vae_scale_factor

    torch.manual_seed(seed)
    latents = torch.randn(
        1, latent_channels, latent_h, latent_w,
        device=pipeline.device,
        dtype=pipeline.dtype
    )

    timestep = torch.tensor([0.5], device=pipeline.device, dtype=pipeline.dtype)

    with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.dtype):
        noise_pred = pipeline.transformer(
            latents,
            timestep,
            encoder_hidden_states=embeddings,
            attention_mask=attention_mask,
        ).sample

    return latents, noise_pred

def test_vae_simple(pipeline, latents):
    # Get VAE decoding
    with torch.no_grad(), torch.autocast("cuda", dtype=pipeline.dtype):
        decoded = pipeline.vae.decode(
            latents / pipeline.vae.config.scaling_factor
        ).sample + pipeline.vae.config.shift_factor
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
    return decoded

# Functions to run some diagnostics
# Basically, I want to know if there are some NaNs/ Infs causing issues
def analyze_tensor(tensor, name):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    # Convert to float32 for accurate statistics
    tensor_f32 = tensor.float()
    min_val = tensor_f32.min().item()
    max_val = tensor_f32.max().item()
    mean_val = tensor_f32.mean().item()
    std_val = tensor_f32.std().item()

    print(f"\n{name}:")
    print(f"  Shape: {list(tensor.shape)}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")

    # Check for issues
    issues = []
    if has_nan or has_inf:
        issues.append("Contains NaN/Inf")
    if abs(mean_val) < 1e-6 and std_val < 1e-6:
        issues.append("Near-zero values (underflow?)")
    if abs(max_val) > 1e4 or abs(min_val) > 1e4:
        issues.append("Very large values (overflow?)")

    if issues:
        print(f"ISSUES: {', '.join(issues)}")
        return True
    return False
