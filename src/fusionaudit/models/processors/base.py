from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_INPUT_KEYS = {
    "pixel_values",
    "pixel_attention_mask",
    "image_grid_thw",
    "image_position_ids",
    "num_soft_tokens_per_image",
    "num_crops",
    "rows",
    "cols",
}


@dataclass(slots=True)
class DiffProcessorInfo:
    model_family: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class DifferentiableProcessor:
    """Base interface for image-space adversarial attacks.

    Implementations take a CHW RGB tensor in [0, 1] and return image tensors
    compatible with the corresponding model's official processor output.
    """

    image_keys: set[str] = IMAGE_INPUT_KEYS

    def __init__(self, model_id: str, official_processor: Any):
        self.model_id = model_id
        self.official_processor = official_processor
        self.info = DiffProcessorInfo(
            model_family="unknown",
            notes="No processor metadata was provided.",
        )

    def preprocess(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        return self.preprocess(image_tensor)

    def describe(self) -> dict[str, str]:
        return self.info.to_dict()


def image_to_float_tensor(
    image: Image.Image,
    *,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div(255.0)
    if device is not None:
        tensor = tensor.to(device)
    tensor.requires_grad_(requires_grad)
    return tensor


def image_to_uint8_tensor(
    image: Image.Image,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    arr = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def project_linf(x_adv: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    delta = torch.clamp(x_adv - x0, min=-eps, max=eps)
    return torch.clamp(x0 + delta, min=0.0, max=1.0)


def random_start(x0: torch.Tensor, eps: float, seed: int | None = None) -> torch.Tensor:
    if seed is None or seed <= 0:
        return x0.detach().clone()
    generator = torch.Generator(device=x0.device)
    generator.manual_seed(seed)
    noise = torch.rand(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
    return project_linf(x0 + (2.0 * noise - 1.0) * eps, x0, eps)


def bounded_uint8_tensor(
    x_adv: torch.Tensor,
    original_uint8: torch.Tensor,
    *,
    eps_int: int = 8,
) -> torch.Tensor:
    rounded = torch.round(x_adv.detach().clamp(0.0, 1.0) * 255.0).to(torch.int16)
    base = original_uint8.to(device=rounded.device, dtype=torch.int16)
    bounded = torch.maximum(torch.minimum(rounded, base + eps_int), base - eps_int)
    return bounded.clamp(0, 255).to(torch.uint8)


def tensor_to_pil(tensor_u8: torch.Tensor) -> Image.Image:
    arr = tensor_u8.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def normalize_chw(
    image: torch.Tensor,
    mean: tuple[float, float, float] | list[float],
    std: tuple[float, float, float] | list[float],
) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean_t) / std_t


def resize_chw(
    image: torch.Tensor,
    size: tuple[int, int],
    *,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    antialias: bool = True,
) -> torch.Tensor:
    return TF.resize(image, list(size), interpolation=interpolation, antialias=antialias)


def center_crop_chw(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return TF.center_crop(image, list(size))


def batch_feature_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported processor output type: {type(value)!r}")


def move_to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value
