"""
Florence-2 + our LoRA adapter for GUI grounding.

Training/inference reference:
    /Users/georgijhabner/Development/ITMO/thesis/vlm_edu/florence/finetune.py
    /Users/georgijhabner/Development/ITMO/thesis/vlm_edu/florence/evaluate.py

Trained with task "<CAPTION_TO_PHRASE_GROUNDING>" + instruction.
Target format: "{instruction}<loc_X><loc_Y><loc_X><loc_Y>" (degenerate bbox = point).
First two <loc_N> tokens are the predicted point; N is in [0, 999] mapped to [0, 1].

Checkpoint path via env var FLORENCE_LORA_PATH — must point to the LoRA adapter
directory (containing adapter_config.json and adapter_model.safetensors).
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


class Florence2LoRA(GUIGroundingModel):
    name = "Florence-2 + LoRA (ours)"
    category: Category = "ours"
    params = "270M"

    def __init__(self, adapter_path: Optional[str] = None):
        path = adapter_path or os.environ.get("FLORENCE_LORA_PATH")
        if not path:
            raise ValueError(
                "Florence2LoRA: set FLORENCE_LORA_PATH env var or pass "
                "adapter_path. Expected path to directory with "
                "adapter_config.json + adapter_model.safetensors."
            )
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Florence2LoRA adapter dir not found: {path}")

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
