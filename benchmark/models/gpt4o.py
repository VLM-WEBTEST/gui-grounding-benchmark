"""
GPT-4o adapter.

Prompt template from ScreenSpot-Pro: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
"""

from .base import GUIGroundingModel, Category


class GPT4o(GUIGroundingModel):
    name = "GPT-4o"
    category: Category = "closed_api"
    params = "N/A"
    cost_per_1k = "$2.50-10.00"

    def predict(self, image, instruction):
        raise NotImplementedError
