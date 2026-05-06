from __future__ import annotations

from PIL import Image


class PromptBuilder:
    """Build Qwen-VL chat-template messages.

    Train messages contain a user turn (image + instruction) followed by an
    assistant turn (target). Inference messages contain only the user turn.
    """

    def __init__(self, instruction: str) -> None:
        if not instruction:
            raise ValueError("instruction must be a non-empty string")
        self.instruction = instruction

    def build_train_messages(self, *, image: Image.Image, target: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.instruction},
                ],
            },
            {"role": "assistant", "content": target},
        ]

    def build_inference_messages(self, *, image: Image.Image) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]
