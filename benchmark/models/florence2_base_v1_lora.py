"""
Florence-2-base + LoRA v1 (older training run, for paper baseline comparison
with v22 on the ec637966 dataset revision).

Same architecture as florence2_lora.py — only the adapter checkpoint differs.
"""

import os
from typing import Optional, Tuple

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from ..parsing.coordinate_parsers import parse_florence_loc_tokens
from .base import Category, GUIGroundingModel

BASE_MODEL = "microsoft/Florence-2-base-ft"
REVISION = "refs/pr/6"
TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Florence2BaseV1LoRA(GUIGroundingModel):
    name = "Florence-2-base + LoRA v1 (ours)"
    category: Category = "ours"
    params = "232M"

    def __init__(self, adapter_path: Optional[str] = None):
        path = adapter_path or os.environ.get("FLORENCE_BASE_V1_LORA_PATH")
        if not path:
            raise ValueError(
                "Florence2BaseV1LoRA: set FLORENCE_BASE_V1_LORA_PATH env var."
            )
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Florence2BaseV1LoRA adapter dir not found: {path}")

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
