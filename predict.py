import os
import cv2
import time
import torch
import subprocess
from PIL import Image
from cog import BasePredictor, Input, Path
from typing import Dict, Optional, Any
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_PATH = "./checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/Qwen/Qwen3-VL-8B-Instruct/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

def get_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """Extract video metadata (fps, duration, frame count) using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            "fps": fps,
            "duration": duration,
            "frame_count": frame_count,
        }
    except Exception as e:
        print(f"Warning: Could not extract video metadata: {e}")
        return None

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load Qwen3-VL Instruct model plus matching processor."""
        # Download weights
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)

        self.model_id = "checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        torch.multiprocessing.set_sharing_strategy("file_system")

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        attn_impl = "flash_attention_2" if self.device.type == "cuda" else None
        model_kwargs = {
            "dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except (TypeError, ValueError, NotImplementedError, RuntimeError):
            # Fallback to default attention if FlashAttention 2 is unavailable.
            model_kwargs.pop("attn_implementation", None)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        self.model.to(self.device)
        self.model.eval()

    def _prepare_inputs(
        self,
        prompt: str,
        image: Optional[Image.Image],
        video_path: Optional[str],
        video_fps: float,
        video_max_pixels: int,
    ) -> Dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]

        # Add image if provided
        if image is not None:
            messages[0]["content"].append({"type": "image", "image": image})

        # Add video if provided
        if video_path is not None:
            video_content = {
                "type": "video",
                "video": video_path,
                "fps": video_fps,
            }
            # Set max_pixels to control video token count
            if video_max_pixels > 0:
                video_content["max_pixels"] = video_max_pixels

            # Extract video metadata to avoid fps warning
            metadata = get_video_metadata(video_path)
            if metadata:
                video_content["video_metadata"] = metadata

            messages[0]["content"].append(video_content)

        messages[0]["content"].append({"type": "text", "text": prompt})

        # Process vision info (images and videos)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Optimize processing based on whether we have video or just images
        if video_path is not None:
            # Video path: use full video metadata support for Qwen3-VL
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            # Unpack video metadata if videos are present
            video_metadatas = None
            if video_inputs is not None and isinstance(video_inputs, list) and len(video_inputs) > 0:
                # When return_video_metadata=True, videos come as (tensor, metadata) tuples
                if isinstance(video_inputs[0], tuple):
                    videos, video_metadatas = zip(*video_inputs)
                    video_inputs = list(videos)
                    video_metadatas = list(video_metadatas)

            # Prepare inputs with processor
            processor_kwargs = {
                "text": [text],
                "images": image_inputs,
                "videos": video_inputs,
                "padding": True,
                "return_tensors": "pt",
            }

            # Add video metadata if available
            if video_metadatas is not None:
                processor_kwargs["video_metadata"] = video_metadatas

            # Add any additional video kwargs
            processor_kwargs.update(video_kwargs)

            batch = self.processor(**processor_kwargs)
        else:
            # Image only: use simpler, faster processing
            image_inputs, video_inputs = process_vision_info(messages)

            batch = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

        batch = batch.to(self.device)

        model_dtype = self.model.dtype
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    batch[key] = value.to(self.device, dtype=model_dtype)
                else:
                    batch[key] = value.to(self.device)

        return batch

    def predict(
        self,
        prompt: str = Input(
            description="Instruction or conversation turn for Qwen3-VL.",
            default="Describe what is happening in the media content",
        ),
        media: Path = Input(
            description="Optional image or video file. Supported formats: images (jpg, png, etc.) and videos (mp4, avi, mov, etc.).",
            default=None,
        ),
        video_fps: float = Input(
            description="Frames per second to sample from video. Only applies to video inputs. Lower values reduce token count.",
            default=1.0,
            ge=0.1,
            le=10.0,
        ),
        video_max_pixels: int = Input(
            description="Maximum pixels for video frames. Only applies to video inputs. Controls token count. Set to 0 for auto. Recommended: 50176-786432.",
            default=200704,
            ge=0,
            le=1048576,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to sample.",
            default=512,
            ge=1,
            le=4096,
        ),
        temperature: float = Input(
            description="Sampling temperature; set to 0 for deterministic output.",
            default=0.7,
            ge=0.0,
            le=2.0,
        ),
        top_p: float = Input(
            description="Cumulative probability for nucleus sampling.",
            default=0.9,
            ge=0.0,
            le=1.0,
        ),
    ) -> str:
        """Run a single multimodal instruction-following generation with optional image or video input."""
        pil_image = None
        video_path = None

        if media is not None:
            media_str = str(media)
            # Detect if input is a video based on file extension
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg')
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

            if media_str.lower().endswith(video_extensions):
                # Process as video
                video_path = media_str
            elif media_str.lower().endswith(image_extensions):
                # Process as image
                with Image.open(media) as pil:
                    pil_image = pil.convert("RGB")
            else:
                # Try to open as image first, fall back to treating as video
                try:
                    with Image.open(media) as pil:
                        pil_image = pil.convert("RGB")
                except Exception:
                    # Assume it's a video
                    video_path = media_str

        # Prepare inputs
        model_inputs = self._prepare_inputs(
            prompt=prompt,
            image=pil_image,
            video_path=video_path,
            video_fps=video_fps,
            video_max_pixels=video_max_pixels,
        )

        # Clean up image
        if pil_image is not None:
            pil_image.close()

        do_sample = temperature > 0

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                do_sample=do_sample,
            )

        prompt_length = model_inputs["input_ids"].shape[-1]
        trimmed_ids = generated_ids[:, prompt_length:]

        outputs = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0].strip()
