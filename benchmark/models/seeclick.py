"""
SeeClick adapter (cckevinn/SeeClick).

Reference:
    Repo:     https://github.com/njucckevin/SeeClick
    Script:   pretrain/screenspot_test.py   (prompt, model load, chat)
    Helpers:  pretrain/process_utils.py     (extract_bbox, pred_2_point)
    HF:       https://huggingface.co/cckevinn/SeeClick

Base model: Qwen-VL-Chat (9.6B). Uses its `.chat()` API — NOT the standard
`.generate()` path — so we go via the tokenizer's `from_list_format` which
expects an image *path*, not a PIL object. We round-trip each image through a
tempfile per call.

Prompt (verbatim from their screenspot_test.py):

    In this UI screenshot, what is the position of the element corresponding
    to the command "{instruction}" (with point)?

Output can be either:
  - a bbox "<box>(x1,y1),(x2,y2)</box>" with integer coords on a 0-1000 grid
    (take center, divide by 1000), or
  - a point as two floats (already normalized to [0, 1]).
We replicate that branching here verbatim from process_utils.py.

Resource: 9.6B + bf16 ≈ 19 GB VRAM. Fits on a 24 GB GPU.
"""

import os
import re
import tempfile
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Category, GUIGroundingModel

MODEL_ID = "cckevinn/SeeClick"
# SeeClick's HF config.json only maps AutoConfig and AutoModelForCausalLM in
# auto_map (no AutoTokenizer). Load the tokenizer from the Qwen-VL-Chat parent
# repo — same vocab, and it ships the AutoTokenizer auto_map entry.
TOKENIZER_ID = "Qwen/Qwen-VL-Chat"

PROMPT_TEMPLATE = (
    'In this UI screenshot, what is the position of the element corresponding '
    'to the command "{instruction}" (with point)?'
)


def _extract_bbox(s: str):
    # Verbatim from SeeClick/pretrain/process_utils.py
    pattern = r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>"
    matches = re.findall(pattern, s)
    return [
        (int(x.split(",")[0]), int(x.split(",")[1]))
        for x in sum(matches, ())
    ]


def _pred_2_point(s: str):
    # Verbatim from SeeClick/pretrain/process_utils.py
    floats = re.findall(r"-?\d+\.?\d*", s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    if len(floats) == 4:
        return [(floats[0] + floats[2]) / 2, (floats[1] + floats[3]) / 2]
    return None


class SeeClick(GUIGroundingModel):
    name = "SeeClick"
    category: Category = "open_gui_specialist"
    params = "9.6B"

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SeeClick requires CUDA — the Qwen-VL custom code uses bf16 and "
                "device_map=cuda and will not run on MPS/CPU."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_ID, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="cuda",
            trust_remote_code=True,
            bf16=True,
        ).eval()

    @torch.no_grad()
    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Qwen-VL's from_list_format requires an image file path.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name
        try:
            query = self._tokenizer.from_list_format([
                {"image": tmp_path},
                {"text": PROMPT_TEMPLATE.format(instruction=instruction)},
            ])
            response, _ = self._model.chat(
                self._tokenizer, query=query, history=None
            )
        finally:
            os.unlink(tmp_path)

        if "box" in response:
            corners = _extract_bbox(response)
            if len(corners) < 2:
                return None
            (x1, y1), (x2, y2) = corners[0], corners[1]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return (cx / 1000, cy / 1000)

        point = _pred_2_point(response)
        if not point:
            return None
        return (float(point[0]), float(point[1]))
