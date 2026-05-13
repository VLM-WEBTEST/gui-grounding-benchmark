"""
WebClick loader (Hcompany/WebClick on HuggingFace).

Independent benchmark of web-UI grounding from Hcompany:
    - 1,639 samples, English-language web screenshots
    - 100+ source websites, 3 buckets:
        agentbrowse (~36%): SurferH agent + WebVoyager tasks
        humanbrowse (~32%): human interactions (shop, travel, organization)
        calendars   (~32%): calendar widget interactions
    - bbox already normalized to [0, 1] as (x_min, y_min, x_max, y_max)

Per the dataset card it was curated with explicit care to NOT overlap with
ScreenSpot / ScreenSpot-V2 (different sources, different annotation methodology).
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple

from datasets import load_dataset
from PIL import Image


@dataclass
class WebClickSample:
    image: Image.Image
    instruction: str
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2) normalized
    bucket: str                                # agentbrowse | humanbrowse | calendars


class WebClick:
    HF_REPO = "Hcompany/WebClick"

    def __init__(self, split: str = "test"):
        self._rows = list(load_dataset(self.HF_REPO, split=split))

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[WebClickSample]:
        for row in self._rows:
            yield self._to_sample(row)

    def __getitem__(self, idx: int) -> WebClickSample:
        return self._to_sample(self._rows[idx])

    @staticmethod
    def _to_sample(row) -> WebClickSample:
        img: Image.Image = row["image"].convert("RGB")
        bbox = tuple(float(v) for v in row["bbox"])
        return WebClickSample(
            image=img,
            instruction=row["instruction"],
            bbox=bbox,
            bucket=str(row.get("bucket", "unknown")),
        )

    def to_lists(self):
        imgs, instrs, bboxes, buckets = [], [], [], []
        for s in self:
            imgs.append(s.image)
            instrs.append(s.instruction)
            bboxes.append(s.bbox)
            buckets.append(s.bucket)
        return imgs, instrs, bboxes, buckets
