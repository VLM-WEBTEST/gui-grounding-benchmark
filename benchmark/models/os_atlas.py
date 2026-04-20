"""
OS-Atlas-Base-7B adapter.

Reference: https://github.com/OS-Copilot/OS-Atlas
HF: OS-Copilot/OS-Atlas-Base-7B
"""

from .base import GUIGroundingModel, Category


class OSAtlas(GUIGroundingModel):
    name = "OS-Atlas-Base-7B"
    category: Category = "open_gui_specialist"
    params = "7B"

    def predict(self, image, instruction):
        raise NotImplementedError
