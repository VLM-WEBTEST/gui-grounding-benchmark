"""
GPT-4o adapter.

Prompt template adapted from the ScreenSpot-Pro GUI grounding baseline:
    https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding

Asks the model for pixel coordinates given a screenshot and a target
instruction, and parses the first "(x, y)" tuple from the response.
If the returned coords look normalized (both ≤ 1) they are used as-is;
otherwise they are divided by the image width/height.

Requires env var OPENAI_API_KEY.
"""

import base64
import io
import os
import re
from typing import Optional, Tuple

from openai import OpenAI
from PIL import Image

from .base import Category, GUIGroundingModel

MODEL = "gpt-4o"

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
    if x <= 1.0 and y <= 1.0:
        # Already normalized.
        return (x, y)
    return (x / w, y / h)


class GPT4o(GUIGroundingModel):
    name = "GPT-4o"
    category: Category = "closed_api"
    params = "N/A"
    cost_per_1k = "$2.50-10.00"

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("GPT4o: set OPENAI_API_KEY env var.")
        self._client = OpenAI(api_key=api_key)

    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Note: empirically, pre-resizing does NOT help GPT-4o (verified on
        # 100 WebClick samples: 4/100 native vs 5/100 resized — within noise).
        # OpenAI tiles large images rather than uniformly downscaling, so the
        # downscaling artifact that hurts Claude doesn't apply here.
        w, h = image.size

        resp = self._client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            max_tokens=40,
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
