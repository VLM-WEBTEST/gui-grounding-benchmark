"""
Moondream2 + our LoRA adapter for GUI grounding.

Training/checkpoint code lives at:
    /Users/georgijhabner/Development/ITMO/thesis/vlm_edu/finetune_lora.py

Load pattern copied here verbatim so this repo has no cross-project import:
    1. Load base Moondream2 at revision 2025-06-21.
    2. Call inner `_setup_caches()` before any `.point()` call.
    3. Monkey-patch `variant_state_dict` in the dynamic module so the LoRA deltas
       are applied when `.point(..., settings={"variant": "custom"})` is requested.
    4. Load the fine-tuned `coord_decoder` weights (saved alongside LoRA).

Checkpoint path via env var MOONDREAM_LORA_PATH — must point to the adapter.pt
file produced by finetune_lora.py's save_checkpoint().
"""

import importlib
import os
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


def _nest(flat: dict) -> dict:
    tree: dict = {}
    for k, v in flat.items():
        parts = k.split(".")
        d = tree
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return tree


def _apply_lora_for_inference(hf_model, checkpoint_path: str) -> None:
    """Load LoRA + coord_decoder and monkey-patch the model's dynamic module."""
    inner = hf_model.model
    pkg = inner.__class__.__module__.rsplit(".", 1)[0]
    dtype = next(hf_model.parameters()).dtype
    device = str(inner.device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    flat = {k: v.to(device=device, dtype=dtype) for k, v in ckpt["lora"].items()}
    lora_dict = _nest(flat)
    importlib.import_module(f"{pkg}.moondream").variant_state_dict = (
        lambda *a, **kw: lora_dict
    )

    if "coord_decoder" in ckpt:
        cd = {
            k.removeprefix("coord_decoder."): v.to(device=device, dtype=dtype)
            for k, v in ckpt["coord_decoder"].items()
        }
        inner.region.coord_decoder.load_state_dict(cd)


class Moondream2LoRA(GUIGroundingModel):
    name = "Moondream2 + LoRA (ours)"
    category: Category = "ours"
    params = "1.86B"

    def __init__(self, checkpoint_path: Optional[str] = None):
        path = checkpoint_path or os.environ.get("MOONDREAM_LORA_PATH")
        if not path:
            raise ValueError(
                "Moondream2LoRA: set MOONDREAM_LORA_PATH env var or pass "
                "checkpoint_path. Expected path to adapter.pt from "
                "finetune_lora.save_checkpoint."
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Moondream2LoRA checkpoint not found: {path}")

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

        _apply_lora_for_inference(self._model, path)

    @torch.no_grad()
    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        result = self._model.model.point(
            image, instruction, settings={"variant": "custom"}
        )
        points = result.get("points") or []
        if not points:
            return None
        p = points[0]
        return (float(p["x"]), float(p["y"]))
