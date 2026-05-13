"""
Qwen2-VL-7B-Instruct adapter.

Reference: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

Prompt format follows the Qwen2-VL grounding convention (also used by the
ScreenSpot-Pro benchmark, https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding):
wrapping the target in <|object_ref_start|>...<|object_ref_end|> reliably
triggers the box-token output format. Without those tokens the model often
falls back to natural-language prose ("The button is at the top right ...").

Output format: '<|box_start|>(x1,y1),(x2,y2)<|box_end|>' with integer coords on
an internal 0-1000 grid (not pixels). Parser converts bbox center to normalized
[0, 1]. See benchmark/parsing/coordinate_parsers.py:parse_qwen2_vl_output.

Resource note: 7B + fp16 ≈ 14 GB. Fits on a 16 GB GPU or Apple Silicon Mac
with ≥ 24 GB unified memory. On CPU-only or MPS with tight memory this will
be very slow (minutes per sample).
"""

from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ..parsing.coordinate_parsers import parse_qwen2_vl_output
from .base import Category, GUIGroundingModel

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# Cap vision tokens to 1280 per image (~ 1M pixels) to avoid OOM on 24 GB GPUs.
# ScreenSpot-Web images are 2560x1440 ≈ 3.7M pixels — at the default cap of 16384
# tokens the vision encoder's attention matrix alone exceeds 20 GiB.
# See https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#api-usage for the
# min_pixels / max_pixels controls.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Qwen2VL(GUIGroundingModel):
    name = "Qwen2-VL-7B"
    category: Category = "open_generalist"
    params = "7B"

    def __init__(self):
        self._device = _pick_device()
        dtype = torch.float32 if self._device == "cpu" else torch.float16

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map={"": self._device} if self._device != "cpu" else None,
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )

    @torch.no_grad()
    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Please provide the bounding box coordinate of "
                            "the region this sentence describes: "
                            f"<|object_ref_start|>{instruction}"
                            f"<|object_ref_end|>"
                        ),
                    },
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self._device)

        generated = self._model.generate(
            **inputs, max_new_tokens=64, do_sample=False
        )
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        out = self._processor.batch_decode(
            trimmed, skip_special_tokens=False
        )[0]

        return parse_qwen2_vl_output(out)
