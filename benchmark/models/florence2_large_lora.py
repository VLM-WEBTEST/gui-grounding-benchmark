"""
Florence-2 large + our LoRA adapter for GUI grounding.

Same architecture and prompt as florence2_lora.py, but using
microsoft/Florence-2-large-ft as the base (770M params vs 232M for base-ft).

Checkpoint path via env var FLORENCE_LARGE_LORA_PATH.
"""

import os
from typing import Optional, Tuple

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from ..parsing.coordinate_parsers import parse_florence_loc_tokens
from .base import Category, GUIGroundingModel

BASE_MODEL = "microsoft/Florence-2-large-ft"
REVISION = "refs/pr/19"
TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Florence2LargeLoRA(GUIGroundingModel):
    name = "Florence-2 large + LoRA (ours)"
    category: Category = "ours"
    params = "770M"

    def __init__(self, adapter_path: Optional[str] = None):
        path = adapter_path or os.environ.get("FLORENCE_LARGE_LORA_PATH")
        if not path:
            raise ValueError(
                "Florence2LargeLoRA: set FLORENCE_LARGE_LORA_PATH env var or pass "
                "adapter_path."
            )
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Florence2LargeLoRA adapter dir not found: {path}")

        self._device = _pick_device()

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            revision=REVISION,
            attn_implementation="sdpa",
        ).to(self._device)
        self._model = PeftModel.from_pretrained(base, path).to(self._device)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            BASE_MODEL, trust_remote_code=True, revision=REVISION,
        )

    @torch.no_grad()
    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = TASK_PROMPT + instruction
        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self._device)

        generated = self._model.generate(
            **inputs, max_new_tokens=50, num_beams=1, do_sample=False,
        )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]

        return parse_florence_loc_tokens(text, take="first")
