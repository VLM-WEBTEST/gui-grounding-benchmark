"""
Unified interface for GUI grounding models.

Each model adapter implements predict() returning a normalized click point.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

from PIL import Image

Category = Literal["closed_api", "open_generalist", "open_gui_specialist", "ours"]


class GUIGroundingModel(ABC):
    """Base class for any model that localizes a UI element on a screenshot."""

    @abstractmethod
    def predict(
        self, image: Image.Image, instruction: str
    ) -> Optional[Tuple[float, float]]:
        """Return predicted click point (x, y) in normalized [0, 1] coordinates,
        or None if the model failed to produce a valid prediction."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name, used in result tables."""
        ...

    @property
    @abstractmethod
    def category(self) -> Category:
        """One of: closed_api, open_generalist, open_gui_specialist, ours."""
        ...

    @property
    def params(self) -> str:
        """Parameter count as a string (e.g. '7B', '232M', 'N/A' for API)."""
        return "N/A"

    @property
    def cost_per_1k(self) -> str:
        """Estimated cost for 1000 inference requests. 'local' for open models."""
        return "local"
