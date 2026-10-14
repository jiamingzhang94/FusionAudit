from __future__ import annotations

import math
from typing import Any

import torch
from torchvision.transforms import InterpolationMode

from fusionaudit.models.processors.base import (
    DiffProcessorInfo,
    DifferentiableProcessor,
    normalize_chw,
    resize_chw,
)


class SmolVLMDiffProcessor(DifferentiableProcessor):
    """SmolVLM2 image splitting in torch.

    Resize, split, pad and normalize as in the official processor, with bicubic
    resizing (torchvision has no differentiable LANCZOS). The split layout is
    computed from the original height and width.
    """

    MAX_IMAGE_SIZE = 4096

    def __init__(self, model_id: str, official_processor: Any):
        super().__init__(model_id, official_processor)
        image_processor = getattr(official_processor, "image_processor", official_processor)
        self.longest_edge = int(getattr(getattr(image_processor, "size", None), "longest_edge", 1536) or 1536)
        max_image_size = getattr(image_processor, "max_image_size", {"longest_edge": 384})
        self.tile_size = int(max_image_size.get("longest_edge", 384))
        self.image_mean = tuple(getattr(image_processor, "image_mean", (0.5, 0.5, 0.5)))
        self.image_std = tuple(getattr(image_processor, "image_std", (0.5, 0.5, 0.5)))
        self.do_image_splitting = bool(getattr(image_processor, "do_image_splitting", True))
        self.do_pad = bool(getattr(image_processor, "do_pad", True))
        self.info = DiffProcessorInfo(
            model_family="SmolVLM2",
            notes="Resize, split, pad and normalize in torch with bicubic resizing.",
        )

    @staticmethod
    def _rescale_to_max_len(height: int, width: int, max_len: int) -> tuple[int, int]:
        aspect_ratio = width / height
        if width >= height:
            width = max_len
            height = int(width / aspect_ratio)
            if height % 2 != 0:
                height += 1
        else:
            height = max_len
            width = int(height * aspect_ratio)
            if width % 2 != 0:
                width += 1
        return max(height, 1), max(width, 1)

    @classmethod
    def _scale_below_upper_bound(cls, height: int, width: int) -> tuple[int, int]:
        aspect_ratio = width / height
        if width >= height and width > cls.MAX_IMAGE_SIZE:
            width = cls.MAX_IMAGE_SIZE
            height = int(width / aspect_ratio)
        elif height > width and height > cls.MAX_IMAGE_SIZE:
            height = cls.MAX_IMAGE_SIZE
            width = int(height * aspect_ratio)
        return max(height, 1), max(width, 1)

    def _initial_resize_size(self, height: int, width: int) -> tuple[int, int]:
        height, width = self._rescale_to_max_len(height, width, self.longest_edge)
        return self._scale_below_upper_bound(height, width)

    def _resize_for_vision_encoder(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        aspect_ratio = width / height
        if width >= height:
            width = math.ceil(width / self.tile_size) * self.tile_size
            height = int(width / aspect_ratio)
            height = math.ceil(height / self.tile_size) * self.tile_size
        else:
            height = math.ceil(height / self.tile_size) * self.tile_size
            width = int(height * aspect_ratio)
            width = math.ceil(width / self.tile_size) * self.tile_size
        return resize_chw(
            image,
            (height, width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _split_images(self, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, height, width = image.shape
        max_h = max_w = self.tile_size
        if height > max_h or width > max_w:
            rows = math.ceil(height / max_h)
            cols = math.ceil(width / max_w)
            crops = []
            for row in range(rows):
                for col in range(cols):
                    top = row * max_h
                    left = col * max_w
                    crops.append(image[:, top : top + max_h, left : left + max_w])
            global_image = resize_chw(
                image,
                (self.tile_size, self.tile_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            return torch.stack(crops + [global_image], dim=0), rows, cols
        return image.unsqueeze(0), 0, 0

    def preprocess(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        _, height, width = image_tensor.shape
        image = resize_chw(
            image_tensor,
            self._initial_resize_size(height, width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        if self.do_image_splitting:
            image = self._resize_for_vision_encoder(image)
            frames, rows, cols = self._split_images(image)
        else:
            square = resize_chw(
                image,
                (self.tile_size, self.tile_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            frames, rows, cols = square.unsqueeze(0), 0, 0

        frames = frames.clamp(0.0, 1.0)
        frames = normalize_chw(frames, self.image_mean, self.image_std)
        pixel_values = frames.unsqueeze(0)
        out: dict[str, Any] = {"pixel_values": pixel_values}
        if self.do_pad:
            out["pixel_attention_mask"] = torch.ones(
                (1, frames.shape[0], self.tile_size, self.tile_size),
                dtype=torch.long,
                device=image_tensor.device,
            )
        out["rows"] = [[rows]]
        out["cols"] = [[cols]]
        return out
