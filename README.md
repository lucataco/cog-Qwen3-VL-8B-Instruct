# Qwen3-VL-8B Instruct Cog Predictor

This repository packages the **Qwen/Qwen3-VL-8B-Instruct** multimodal model for [Replicate's Cog](https://github.com/replicate/cog) runtime. The predictor accepts a text prompt with optional image or video input and returns the model's generated response.

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

### Image Understanding
Generate text conditioned on an image:
```bash
cog predict -i prompt="Describe the image." -i media=@/path/to/image.jpg
```

### Video Understanding
Analyze video content:
```bash
cog predict -i prompt="What is happening in this video?" -i media=@/path/to/video.mp4
```

Control video processing parameters:
```bash
cog predict \
  -i prompt="Describe the key events in this video." \
  -i media=@/path/to/video.mp4 \
  -i video_fps=2.0 \
  -i video_max_pixels=400000
```

### Text-Only Inference
Generate text without visual input:
```bash
cog predict -i prompt="Write a haiku about model distillation."
```

### Key Parameters
- `media` – Path to image or video file (auto-detects format)
- `prompt` (default: "Describe this content.") – Text instruction for the model
- `video_fps` (default: 1.0) – Frames per second to sample from video; lower values reduce token count
- `video_max_pixels` (default: 200704) – Maximum pixels for video frames; controls token count for videos
- `max_new_tokens` (default: 512) – Upper bound on generated tokens
- `temperature` (default: 0.7) – Sampling temperature; set to `0` for deterministic output
- `top_p` (default: 0.9) – Cumulative probability threshold for nucleus sampling

## Implementation Notes
- **Video Support**: The model supports video understanding through the `qwen-vl-utils` library. Videos are automatically processed by sampling frames at the specified FPS rate.
- **Video Metadata**: OpenCV (`opencv-python`) is used to extract video metadata (fps, duration, frame count) for accurate frame sampling.
- **Supported Formats**:
  - Images: jpg, jpeg, png, gif, bmp, tiff, webp
  - Videos: mp4, avi, mov, mkv, flv, wmv, webm, m4v, mpeg, mpg
- **Video Processing**: The `video_fps` and `video_max_pixels` parameters control how many frames are extracted and their resolution, directly affecting token count and memory usage.
- **FlashAttention**: Automatic FlashAttention 2 support is enabled when running on CUDA; the predictor falls back to PyTorch's default attention kernels if FlashAttention is unavailable.
- **Precision**: On GPUs, tensors are cast to `bfloat16` for improved throughput. On CPU, the predictor uses `float32`.
- The environment variable `MP_NO_RESOURCE_TRACKER` is set to avoid spurious semaphore leak warnings when using PyTorch multiprocessing.

## Troubleshooting
- **Out of memory**:
  - For images: Reduce `max_new_tokens` or use lower resolution images
  - For videos: Lower `video_fps`, reduce `video_max_pixels`, or use shorter video clips
  - Ensure the GPU meets the 8B parameter model requirements
- **Video processing errors**: Ensure video file is valid and in a supported format. Try converting to MP4 if issues persist.
- **FlashAttention import errors**: FlashAttention is optional. The predictor will retry without it, so no manual intervention is necessary unless you want the performance boost.
- **Slow startup**: The first run downloads ~16 GB of weights. Subsequent runs use the cached files under `checkpoints/`.

## Extending
To customize behavior:
- Adjust prompt preprocessing or response postprocessing in `Predictor._prepare_inputs` and `Predictor.predict`.
- Add new CLI inputs by extending the signature of `Predictor.predict` with additional `cog.Input` definitions.

