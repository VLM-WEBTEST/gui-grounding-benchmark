"""
Loader for the ScreenSpot benchmark (Cheng et al., ACL 2024).

Source: https://github.com/njucckevin/SeeClick
HuggingFace mirror: rootsautomation/ScreenSpot (https://huggingface.co/datasets/rootsautomation/ScreenSpot)

We use only the web subset, split into "text" and "icon" types.
"""

from dataclasses import dataclass
from typing import Iterator, List, Literal, Tuple

from datasets import load_dataset
from PIL import Image

DataType = Literal["text", "icon"]


@dataclass
class ScreenSpotSample:
    image: Image.Image           # PIL RGB
    instruction: str             # natural-language target
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max), normalized [0, 1]
    data_type: DataType          # "text" or "icon"


class ScreenSpotWeb:
    """ScreenSpot web subset, normalized bboxes, lazy PIL images."""

    HF_REPO = "rootsautomation/ScreenSpot"

    def __init__(self, split: str = "test"):
        raw = load_dataset(self.HF_REPO, split=split)
        # ScreenSpot stores platform in "file_name" or a dedicated "platform" column;
        # HF mirrors vary. Filter to the web platform:
        filtered = [row for row in raw if self._is_web(row)]
        self._rows = filtered

    @staticmethod
    def _is_web(row) -> bool:
        # Different mirrors use different field names. Try common candidates.
        for key in ("platform", "data_source", "source"):
            if key in row and str(row[key]).lower().startswith("web"):
                return True
        # Fall back: filename prefix "web_"
        for key in ("file_name", "image_file"):
            if key in row and str(row[key]).lower().startswith("web"):
                return True
        return False

    @staticmethod
    def _normalize_bbox(bbox) -> Tuple[float, float, float, float]:
        """rootsautomation/ScreenSpot bboxes are already (x1, y1, x2, y2) normalized to [0, 1]."""
        x1, y1, x2, y2 = bbox
        return (float(x1), float(y1), float(x2), float(y2))

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[ScreenSpotSample]:
        for row in self._rows:
            yield self._to_sample(row)

    def __getitem__(self, idx: int) -> ScreenSpotSample:
        return self._to_sample(self._rows[idx])

    def _to_sample(self, row) -> ScreenSpotSample:
        image: Image.Image = row["image"].convert("RGB")
        bbox = self._normalize_bbox(row["bbox"])
        data_type = row.get("data_type", "text")
        if data_type not in ("text", "icon"):
            data_type = "icon"
        return ScreenSpotSample(
            image=image,
            instruction=row["instruction"],
            bbox=bbox,
            data_type=data_type,
        )

    def to_lists(self) -> Tuple[List[Image.Image], List[str], List[tuple], List[str]]:
        """Materialize all samples into parallel lists. Use for small datasets only."""
        images, instructions, bboxes, types = [], [], [], []
        for s in self:
            images.append(s.image)
            instructions.append(s.instruction)
            bboxes.append(s.bbox)
            types.append(s.data_type)
        return images, instructions, bboxes, types
