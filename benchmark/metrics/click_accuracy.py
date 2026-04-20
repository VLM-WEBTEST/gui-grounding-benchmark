"""
ClickAcc metric — ScreenSpot / SeeClick protocol.

Reference: Cheng et al., "SeeClick: Harnessing GUI Grounding for Advanced Visual
GUI Agents", ACL 2024. https://arxiv.org/abs/2401.10935

A prediction is counted as correct iff the predicted point (x, y) falls inside
the ground-truth bounding box [x_min, y_min, x_max, y_max]. All coordinates
normalized to [0, 1].

Overall ClickAcc = correct / total
ClickAcc-Text    = correct / total over samples with data_type == "text"
ClickAcc-Icon    = correct / total over samples with data_type == "icon" (a.k.a. Icon/Widget)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def point_in_bbox(point: Point, bbox: BBox) -> bool:
    if point is None or bbox is None:
        return False
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max


def click_accuracy(
    preds: List[Optional[Point]],
    gt_bboxes: List[BBox],
    data_types: List[str],
) -> Dict[str, float]:
    """Compute ClickAcc Overall / Text / Icon.

    Args:
        preds: list of (x, y) points in [0, 1], or None for failed predictions.
        gt_bboxes: list of [x_min, y_min, x_max, y_max] in [0, 1].
        data_types: list of "text" or "icon" per sample.

    Returns: {"overall": float, "text": float, "icon": float,
              "n_overall": int, "n_text": int, "n_icon": int,
              "n_failed": int}
    """
    assert len(preds) == len(gt_bboxes) == len(data_types), "mismatched lengths"

    hits = [point_in_bbox(p, bb) for p, bb in zip(preds, gt_bboxes)]
    n = len(hits)
    n_failed = sum(1 for p in preds if p is None)

    text_idx = [i for i, t in enumerate(data_types) if t == "text"]
    icon_idx = [i for i, t in enumerate(data_types) if t == "icon"]

    def _acc(indices):
        if not indices:
            return float("nan")
        return float(np.mean([hits[i] for i in indices]))

    return {
        "overall": float(np.mean(hits)) if hits else float("nan"),
        "text":    _acc(text_idx),
        "icon":    _acc(icon_idx),
        "n_overall": n,
        "n_text":    len(text_idx),
        "n_icon":    len(icon_idx),
        "n_failed":  n_failed,
    }
