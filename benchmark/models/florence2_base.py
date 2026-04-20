"""
Florence-2 base adapter.

Uses microsoft/Florence-2-base-ft (same base as our LoRA) so the benchmark
isolates the LoRA delta cleanly. Task prompt matches our LoRA's training task
(<CAPTION_TO_PHRASE_GROUNDING>) so base vs LoRA is an apples-to-apples comparison
on the same task. The alternative OVD task returns polygons for GUI-style
instructions ("create a new project") rather than bboxes — unsuitable here.

Output parse: processor.post_process_generation returns
{"<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [[x1, y1, x2, y2], ...], "labels": [...]}}
in pixel coordinates. We take the first bbox center, normalized by image size.

Reference: https://huggingface.co/microsoft/Florence-2-base-ft
"""

from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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


class Florence2Base(GUIGroundingModel):
    name = "Florence-2 (base)"
    category: Category = "open_generalist"
    params = "270M"

    def __init__(self):
        self._device = _pick_device()

        self._model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            revision=REVISION,
            attn_implementation="sdpa",
        ).to(self._device)
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
        parsed = self._processor.post_process_generation(
            text, task=TASK_PROMPT, image_size=(image.width, image.height)
        )

        bboxes = parsed.get(TASK_PROMPT, {}).get("bboxes") or []
        if not bboxes:
            return None
        x1, y1, x2, y2 = bboxes[0]
        cx = (x1 + x2) / 2 / image.width
        cy = (y1 + y2) / 2 / image.height
        return (float(cx), float(cy))
