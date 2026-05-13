"""
OS-Atlas-Base-7B adapter.

Reference: https://github.com/OS-Copilot/OS-Atlas
HF:        https://huggingface.co/OS-Copilot/OS-Atlas-Base-7B

Architecture: Qwen2-VL-7B fine-tuned on GUI screenshots. Output format matches
Qwen2-VL: '<|box_start|>(x1,y1),(x2,y2)<|box_end|>' with integer coords on an
internal 0-1000 grid. Parser converts bbox center to normalized [0, 1].

Prompt copied verbatim from OS-Atlas grounding examples. The specific wording
("In the screenshot of this web page, please give me the coordinates ... in
x1y1x2y2 format.\\n{instruction}") is what their eval scripts use.

Resource: 7B + fp16 ≈ 14 GB VRAM. Fits on a 24 GB GPU.
"""

from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ..parsing.coordinate_parsers import parse_qwen2_vl_output
from .base import Category, GUIGroundingModel

MODEL_ID = "OS-Copilot/OS-Atlas-Base-7B"

# Same vision-token cap as Qwen2-VL (OS-Atlas is a Qwen2-VL fine-tune).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

PROMPT_TEMPLATE = (
    "In the screenshot of this web page, please give me the coordinates of "
    "the element I want to click on according to my instructions (in x1y1x2y2 "
    "format).\n{instruction}"
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OSAtlas(GUIGroundingModel):
    name = "OS-Atlas-Base-7B"
    category: Category = "open_gui_specialist"
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
                        "text": PROMPT_TEMPLATE.format(instruction=instruction),
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
