"""
Claude 3.5 Sonnet adapter.

Prompt template from ScreenSpot-Pro: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
"""

from .base import GUIGroundingModel, Category


class ClaudeSonnet(GUIGroundingModel):
    name = "Claude 3.5 Sonnet"
    category: Category = "closed_api"
    params = "N/A"
    cost_per_1k = "$3.00-15.00"

    def predict(self, image, instruction):
        raise NotImplementedError
