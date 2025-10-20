import os
import time
import torch
import subprocess
from typing import Dict, Optional
from cog import BasePredictor, Input, Path
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "./checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/Qwen/Qwen3-VL-8B-Instruct/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

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
            "torch_dtype": torch_dtype,
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

    def _prepare_inputs(self, prompt: str, image: Optional[Image.Image]) -> Dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]

        if image is not None:
            messages[0]["content"].append({"type": "image", "image": image})
        messages[0]["content"].append({"type": "text", "text": prompt})

        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
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
        ),
        image: Optional[Path] = Input(
            description="Optional image supplying visual context.",
            default=None,
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
        """Run a single multimodal instruction-following generation."""
        pil_image: Optional[Image.Image] = None
        if image is not None:
            with Image.open(image) as pil:
                pil_image = pil.convert("RGB")

        model_inputs = self._prepare_inputs(prompt, pil_image)
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
