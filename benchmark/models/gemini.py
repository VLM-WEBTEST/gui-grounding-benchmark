"""
Gemini 2.5 Pro adapter (via VSellm OpenAI-compatible proxy).

Same prompt template as GPT-4o for direct comparison. Uses the OpenAI SDK with
`base_url=https://api.vsellm.ru/v1` and model id `google/gemini-2.5-pro`.

Requires env var OPENAI_API_KEY (proxy supports both OpenAI and Google routes).
"""

import base64
import io
import os
import re
from typing import Optional, Tuple

from openai import OpenAI
from PIL import Image

from .base import Category, GUIGroundingModel

MODEL = "google/gemini-2.5-pro"

PROMPT_TEMPLATE = (
    'In this UI screenshot, what are the pixel coordinates (x, y) of the '
    'element corresponding to the following instruction: "{instruction}". '
    'Image size: {w}x{h}. Answer with only (x, y).'
)


def _encode_png_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _parse_xy(text: str, w: int, h: int) -> Optional[Tuple[float, float]]:
    m = re.search(r"[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]", text)
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    # Already in [0, 1].
    if x <= 1.0 and y <= 1.0:
        return (x, y)
    # Gemini natively uses 0-1000 normalized coordinates (per Google docs:
    # "coordinates relative to image dimensions, scale to [0, 1000]"). Even
    # when prompted for pixel coordinates, it often returns 0-1000. We treat
    # any value ≤ 1000 as 0-1000 normalized; only fall back to pixels when
    # coords clearly exceed 1000 (which indicates true pixel output).
    if x <= 1000 and y <= 1000:
        return (x / 1000, y / 1000)
    return (x / w, y / h)


class Gemini(GUIGroundingModel):
    name = "Gemini 2.5 Pro"
    category: Category = "closed_api"
    params = "N/A"
    cost_per_1k = "$1.25-10.00"

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Gemini: set OPENAI_API_KEY env var.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size

        resp = self._client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            # Gemini 2.5 Pro reserves at least 128 tokens for internal "thinking"
            # so max_tokens must be >= ~256 (proxy derives thinking budget as
            # half of max_tokens). Visible output is short ("(x, y)").
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT_TEMPLATE.format(
                                instruction=instruction, w=w, h=h
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _encode_png_data_url(image)},
                        },
                    ],
                }
            ],
        )
        text = resp.choices[0].message.content or ""
        return _parse_xy(text, w, h)
