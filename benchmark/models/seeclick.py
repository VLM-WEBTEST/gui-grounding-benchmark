"""
SeeClick adapter.

Reference: https://github.com/njucckevin/SeeClick
Use screenspot_test.py from their repo as reference for prompt and parsing.
Base: Qwen-VL (9.6B).
"""

from .base import GUIGroundingModel, Category


class SeeClick(GUIGroundingModel):
    name = "SeeClick"
    category: Category = "open_gui_specialist"
    params = "9.6B"

    def predict(self, image, instruction):
        raise NotImplementedError
