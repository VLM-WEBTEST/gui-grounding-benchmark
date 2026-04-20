"""
Moondream2 base adapter.

Uses the same revision as our LoRA (2025-06-21) so the benchmark isolates the
LoRA delta cleanly. Calls the native `.point()` API with no `settings` override,
which routes through the model's default variant (no variant_state_dict injection).

Reference: https://huggingface.co/vikhyatk/moondream2
"""

from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from .base import Category, GUIGroundingModel

MD_REVISION = "2025-06-21"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Moondream2Base(GUIGroundingModel):
    name = "Moondream2 (base)"
    category: Category = "open_generalist"
    params = "1.86B"

    def __init__(self):
        self._device = _pick_device()
        dtype = torch.float32 if self._device == "cpu" else torch.bfloat16

        self._model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=MD_REVISION,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": self._device} if self._device != "cpu" else None,
        )
        self._model.model._setup_caches()
        self._model.eval()

    @torch.no_grad()
    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        # Pass ScreenSpot instruction verbatim as the `object` argument.
        # Do NOT rephrase (e.g. "Locate the button labeled ...") — training-time
        # prompts used short phrases and rewording degrades base performance.
        result = self._model.model.point(image, instruction)
        points = result.get("points") or []
        if not points:
            return None
        p = points[0]
        return (float(p["x"]), float(p["y"]))
