# Qwen3-VL-8B Instruct Cog Predictor

This repository packages the **Qwen/Qwen3-VL-8B-Instruct** multimodal model for [Replicate's Cog](https://github.com/replicate/cog) runtime. The predictor accepts a text prompt with optional image context and returns the model's generated response.

## Prerequisites
- CUDA-capable GPU with compute support for CUDA 12.1
- Python 3.10 (managed by Cog inside the Docker image)
- Docker installed locally if you plan to run the containerized Cog runtime

## Project Layout
- `predict.py` – Predictor implementation executed by Cog
- `cog.yaml` – Build configuration describing dependencies and runtime settings
- `requirements.txt` – Python packages installed during the Cog build
- `checkpoints/` – Local cache directory for model weights (created on first run)

## Setup
1. Install Cog:
   ```bash
   pip install cog
   ```
2. Pull the model weights the first time you run the predictor. Cog and `transformers` will automatically download them into the `checkpoints/` directory under the workspace.

## Usage
Generate text conditioned on an optional image:
```bash
cog predict -i prompt="Describe the image." -i image=@/path/to/image.jpg
```

Text-only inference:
```bash
cog predict -i prompt="Write a haiku about model distillation."
```

Key runtime flags:
- `max_new_tokens` (default 512) – Upper bound on generated tokens
- `temperature` (default 0.7) – Sampling temperature; set to `0` for deterministic output
- `top_p` (default 0.9) – Cumulative probability threshold for nucleus sampling

## Implementation Notes
- Automatic FlashAttention 2 support is enabled when running on CUDA; the predictor falls back to PyTorch's default attention kernels if FlashAttention is unavailable.
- On GPUs, tensors are cast to `bfloat16` for improved throughput. On CPU, the predictor uses `float32`.
- The environment variable `MP_NO_RESOURCE_TRACKER` is set to avoid spurious semaphore leak warnings when using PyTorch multiprocessing.

## Troubleshooting
- **Out of memory**: Reduce `max_new_tokens` or switch to a smaller prompt. Ensure the GPU meets the 8B parameter model requirements.
- **FlashAttention import errors**: FlashAttention is optional. The predictor will retry without it, so no manual intervention is necessary unless you want the performance boost.
- **Slow startup**: The first run downloads ~16 GB of weights. Subsequent runs use the cached files under `checkpoints/`.

## Extending
To customize behavior:
- Adjust prompt preprocessing or response postprocessing in `Predictor._prepare_inputs` and `Predictor.predict`.
- Add new CLI inputs by extending the signature of `Predictor.predict` with additional `cog.Input` definitions.

