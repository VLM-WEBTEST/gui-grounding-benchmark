"""
Claude Sonnet adapter (via the user's VSellm proxy, model id
"anthropic/claude-sonnet-4.6").

Plain-prompt approach mirroring the GPT-4o adapter: ask for pixel coordinates
as "(x, y)" and parse the first tuple. Prompt adapted from ScreenSpot-Pro:
    https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding

The Claude-native Computer Use tool (computer_20250124) would likely score
higher on GUI grounding, but the VSellm proxy routes through litellm which
rejects the Anthropic tool type ("tools.0.custom.input_schema.type: Input
should be 'object'") — so Computer Use is not usable via this proxy.
Plain prompt it is, which also keeps Claude directly comparable with GPT-4o.

Requires env vars ANTHROPIC_API_KEY (+ ANTHROPIC_BASE_URL for the proxy).
"""

import base64
import io
import os
import re
from typing import Optional, Tuple

from anthropic import Anthropic
from PIL import Image

from .base import Category, GUIGroundingModel

MODEL = "anthropic/claude-sonnet-4.6"

# Anthropic API downscales images larger than ~1568px on the long side before
# the vision encoder sees them. If we send the full 2560x1440 ScreenSpot image
# but tell Claude "image size 2560x1440", it reasons in the *downscaled* frame
# and returns coords there — which our parser de-normalizes by the original
# dimensions, biasing predictions toward the top-left and giving artificially
# low ClickAcc (~2-3%). Pre-resizing on our side and reporting the post-resize
# size keeps both sides in the same coordinate frame.
# Verified: WebClick (images already <1568px, no resize triggered) → 92% acc.
MAX_LONG_SIDE = 1568

PROMPT_TEMPLATE = (
    'In this UI screenshot, what are the pixel coordinates (x, y) of the '
    'element corresponding to the following instruction: "{instruction}". '
    'Image size: {w}x{h}. Answer with only (x, y).'
)


def _resize_for_api(image: Image.Image) -> Image.Image:
    w, h = image.size
    if max(w, h) <= MAX_LONG_SIDE:
        return image
    if w >= h:
        new_w = MAX_LONG_SIDE
        new_h = round(h * MAX_LONG_SIDE / w)
    else:
        new_h = MAX_LONG_SIDE
        new_w = round(w * MAX_LONG_SIDE / h)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _encode_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_xy(text: str, w: int, h: int) -> Optional[Tuple[float, float]]:
    m = re.search(r"[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]", text)
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    if x <= 1.0 and y <= 1.0:
        return (x, y)
    return (x / w, y / h)


class ClaudeSonnet(GUIGroundingModel):
    name = "Claude Sonnet 4.6"
    category: Category = "closed_api"
    params = "N/A"
    cost_per_1k = "$3.00-15.00"

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ClaudeSonnet: set ANTHROPIC_API_KEY env var.")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        self._client = (
            Anthropic(api_key=api_key, base_url=base_url) if base_url
            else Anthropic(api_key=api_key)
        )

    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = _resize_for_api(image)
        w, h = image.size

        resp = self._client.messages.create(
            model=MODEL,
            max_tokens=64,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _encode_png_b64(image),
                            },
                        },
                        {
                            "type": "text",
                            "text": PROMPT_TEMPLATE.format(
                                instruction=instruction, w=w, h=h
                            ),
                        },
                    ],
                }
            ],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        return _parse_xy(text, w, h)
