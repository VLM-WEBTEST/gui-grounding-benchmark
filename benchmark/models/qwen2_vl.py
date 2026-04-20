"""
Qwen2-VL-7B-Instruct adapter.

Reference: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
Prompt format and coordinate post-processing per model card.
"""

from .base import GUIGroundingModel, Category


class Qwen2VL(GUIGroundingModel):
    name = "Qwen2-VL-7B"
    category: Category = "open_generalist"
    params = "7B"

    def predict(self, image, instruction):
        raise NotImplementedError
