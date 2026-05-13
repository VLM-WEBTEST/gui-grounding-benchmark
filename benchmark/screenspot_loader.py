"""
Loader for ScreenSpot-V2 (Cheng et al., updated for OS-Atlas; lmms-lab/ScreenSpot-v2).

V2 differences vs V1 (rootsautomation/ScreenSpot):
  - bbox format is [x, y, w, h] in *pixels* (V1 was [x1, y1, x2, y2] normalized)
  - single split named "train" (V1 used "test")
  - 1272 total samples; web subset (file_name prefix "web_") = 437

We benchmark only the web subset, split into "text" and "icon" types.
"""

from dataclasses import dataclass
from typing import Iterator, Literal, Tuple

from datasets import load_dataset
from PIL import Image

DataType = Literal["text", "icon"]


@dataclass
class ScreenSpotSample:
    image: Image.Image                            # PIL RGB
    instruction: str                              # natural-language target
    bbox: Tuple[float, float, float, float]       # (x1, y1, x2, y2), normalized [0, 1]
    data_type: DataType                           # "text" or "icon"


class ScreenSpotWeb:
    """ScreenSpot-V2 web subset, normalized bboxes, lazy PIL images."""

    HF_REPO = "lmms-lab/ScreenSpot-v2"

    def __init__(self, split: str = "train"):
        raw = load_dataset(self.HF_REPO, split=split)
        self._rows = [row for row in raw if self._is_web(row)]

    @staticmethod
    def _is_web(row) -> bool:
        fn = row.get("img_filename") or row.get("file_name") or ""
        return str(fn).lower().startswith("web_")

    @staticmethod
    def _normalize_bbox(bbox_xywh, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """V2 bbox is [x, y, w, h] in pixels. Convert to (x1, y1, x2, y2) normalized [0, 1]."""
        x, y, w, h = bbox_xywh
        return (
            float(x) / img_w,
            float(y) / img_h,
            float(x + w) / img_w,
            float(y + h) / img_h,
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[ScreenSpotSample]:
        for row in self._rows:
            yield self._to_sample(row)

    def __getitem__(self, idx: int) -> ScreenSpotSample:
        return self._to_sample(self._rows[idx])

    def _to_sample(self, row) -> ScreenSpotSample:
        image: Image.Image = row["image"].convert("RGB")
        bbox = self._normalize_bbox(row["bbox"], image.width, image.height)
        data_type = row.get("data_type", "text")
        if data_type not in ("text", "icon"):
            data_type = "icon"
        return ScreenSpotSample(
            image=image,
            instruction=row["instruction"],
            bbox=bbox,
            data_type=data_type,
        )
