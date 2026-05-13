"""
Khabner/moondream-data loader (in-domain test set for our LoRA models).

This is the dataset our Moondream2/Florence-2 LoRA models were trained on
(via the train split). The held-out test split (10% = 2,180 samples) is used
here for in-domain evaluation, with caveats:

  - Our LoRA models had access to TRAIN split during fine-tuning. Evaluating
    them on TEST is honest held-out (no leakage between splits) but still
    in-distribution.
  - All other models (proprietary APIs, base open models, GUI specialists)
    are evaluated zero-shot — they have not seen this distribution.

Schema:
    image: PIL (variable size, 1280x720 / 1440x900 / 1920x1080)
    instruction: str (6-88 chars)
    point: {"x": float, "y": float} normalized [0, 1]
    bbox: {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
          normalized [0, 1]
    element_type: str (button|input|link|label|checkbox|heading|menu_item|textarea|...)
    element_label: str (1-60 chars)
    site_name: str (40 categories, e.g. crm_system, fintech_app, travel_site)
    url: str
    viewport: str (e.g. "1920x1080")
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple

from datasets import load_dataset
from PIL import Image


@dataclass
class KhabnerSample:
    image: Image.Image
    instruction: str
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2) normalized
    element_type: str
    site_name: str
    viewport: str


class KhabnerWebTest:
    HF_REPO = "Khabner/moondream-data"
    # Pin to a specific revision for reproducibility. ec637966 is the revision
    # used to train moondream-lora v12 and florence-large-lora v1 — eval on its
    # test split is in-domain held-out for those LoRAs.
    REVISION = "ec637966"

    def __init__(self, split: str = "test", revision: str = None):
        rev = revision or self.REVISION
        self._rows = list(load_dataset(self.HF_REPO, split=split, revision=rev))

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[KhabnerSample]:
        for row in self._rows:
            yield self._to_sample(row)

    def __getitem__(self, idx: int) -> KhabnerSample:
        return self._to_sample(self._rows[idx])

    @staticmethod
    def _to_sample(row) -> KhabnerSample:
        img: Image.Image = row["image"].convert("RGB")
        bb = row["bbox"]
        bbox = (
            float(bb["x_min"]),
            float(bb["y_min"]),
            float(bb["x_max"]),
            float(bb["y_max"]),
        )
        return KhabnerSample(
            image=img,
            instruction=row["instruction"],
            bbox=bbox,
            element_type=str(row.get("element_type", "unknown")),
            site_name=str(row.get("site_name", "unknown")),
            viewport=str(row.get("viewport", "unknown")),
        )

    def to_lists(self):
        imgs, instrs, bboxes, types, sites = [], [], [], [], []
        for s in self:
            imgs.append(s.image)
            instrs.append(s.instruction)
            bboxes.append(s.bbox)
            types.append(s.element_type)
            sites.append(s.site_name)
        return imgs, instrs, bboxes, types, sites
