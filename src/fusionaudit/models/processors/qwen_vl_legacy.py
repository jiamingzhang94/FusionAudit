from __future__ import annotations

from typing import Any

import torch
from torchvision.transforms import InterpolationMode

from fusionaudit.models.processors.base import (
    DiffProcessorInfo,
    DifferentiableProcessor,
    normalize_chw,
    resize_chw,
)


class LegacyQwenVLDiffProcessor(DifferentiableProcessor):
    """Differentiable replacement for Qwen-VL-Chat's visual.image_transform."""

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_id: str, official_processor: Any):
        super().__init__(model_id, official_processor)
        self.image_size = self._infer_image_size(official_processor)
        self.info = DiffProcessorInfo(
            model_family="Qwen-VL-Chat",
            notes=(
                "Reimplements visual.image_transform as bicubic resize to the "
                "model image_size followed by CLIP normalization."
            ),
        )

    @staticmethod
    def _infer_image_size(processor: Any) -> int:
        model = getattr(processor, "model", None)
        visual = None
        try:
            visual = model.transformer.visual
        except Exception:
            pass
        image_size = getattr(visual, "image_size", 448)
        if isinstance(image_size, (tuple, list)):
            return int(image_size[0])
        return int(image_size)

    def preprocess(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        resized = resize_chw(
            image_tensor,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        resized = resized.clamp(0.0, 1.0)
        pixel_values = normalize_chw(resized, self.mean, self.std).unsqueeze(0)
        return {"pixel_values": pixel_values}
