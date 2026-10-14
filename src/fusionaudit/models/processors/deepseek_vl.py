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


class DeepSeekVLDiffProcessor(DifferentiableProcessor):
    """Torch replacement for DeepSeek-VL-7B's PIL/numpy image processor."""

    def __init__(self, model_id: str, official_processor: Any):
        super().__init__(model_id, official_processor)
        image_processor = getattr(official_processor, "image_processor", official_processor)
        self.image_size = int(getattr(image_processor, "image_size", 1024))
        self.min_size = int(getattr(image_processor, "min_size", 14))
        self.image_mean = tuple(getattr(image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073)))
        self.image_std = tuple(getattr(image_processor, "image_std", (0.26862954, 0.26130258, 0.27577711)))
        self.do_normalize = bool(getattr(image_processor, "do_normalize", True))
        self.info = DiffProcessorInfo(
            model_family="DeepSeek-VL-7B",
            notes=(
                "Reimplements resize-by-long-edge, square padding with official "
                "background color, rescale, and normalization in torch."
            ),
        )

    def _resize_size(self, height: int, width: int) -> tuple[int, int]:
        max_size = max(height, width)
        new_h = max(int(height / max_size * self.image_size), self.min_size)
        new_w = max(int(width / max_size * self.image_size), self.min_size)
        return new_h, new_w

    def _expand_to_square(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        if height == width:
            return image
        side = max(height, width)
        bg = torch.tensor(
            [int(v * 255) / 255.0 for v in self.image_mean],
            dtype=image.dtype,
            device=image.device,
        ).view(3, 1, 1)
        canvas = bg.expand(3, side, side).clone()
        top = (side - height) // 2
        left = (side - width) // 2
        canvas[:, top : top + height, left : left + width] = image
        return canvas

    def preprocess(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        _, height, width = image_tensor.shape
        resized = resize_chw(
            image_tensor,
            self._resize_size(height, width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        resized = resized.clamp(0.0, 1.0)
        squared = self._expand_to_square(resized)
        if self.do_normalize:
            squared = normalize_chw(squared, self.image_mean, self.image_std)
        # DeepSeek-VL expects [B, N, C, H, W] where N=number of image crops
        return {"pixel_values": squared.unsqueeze(0).unsqueeze(0)}
