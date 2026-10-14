from __future__ import annotations

from typing import Any

import torch

from fusionaudit.models.processors.base import (
    DiffProcessorInfo,
    DifferentiableProcessor,
    batch_feature_to_dict,
)


class OfficialTorchDiffProcessor(DifferentiableProcessor):
    """Use the official HF Torchvision-backed image processor with gradients."""

    def __init__(
        self,
        model_id: str,
        official_processor: Any,
        *,
        model_family: str,
        notes: str,
    ):
        super().__init__(model_id, official_processor)
        image_processor = getattr(official_processor, "image_processor", official_processor)
        if image_processor is None:
            raise ValueError(f"{model_id} has no image_processor")
        self.image_processor = image_processor
        self.info = DiffProcessorInfo(
            model_family=model_family,
            notes=notes,
        )

    def _clamp_normalized(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Clamp pixel_values to the valid normalized range.

        Bicubic interpolation on float tensors can overshoot [0, 1] before
        normalization.  The official PIL path avoids this because uint8 is
        naturally bounded.  We replicate that bound by clamping to the
        normalized equivalents of 0.0 and 1.0.
        """
        ip = self.image_processor
        if not (hasattr(ip, "image_mean") and hasattr(ip, "image_std")
                and getattr(ip, "do_normalize", True)):
            return pixel_values

        mean = torch.tensor(ip.image_mean, dtype=pixel_values.dtype,
                            device=pixel_values.device)
        std = torch.tensor(ip.image_std, dtype=pixel_values.dtype,
                           device=pixel_values.device)
        pv_min = (0.0 - mean) / std
        pv_max = (1.0 - mean) / std

        ndim = pixel_values.ndim
        if ndim >= 3:
            # Standard [B,C,H,W] or [B,N,C,H,W]: channel at dim -3
            shape = [1] * ndim
            shape[-3] = 3
            pv_min = pv_min.view(*shape)
            pv_max = pv_max.view(*shape)
        elif ndim == 2:
            # Qwen-style packed [num_patches, flat_dim]:
            # flat_dim = C * temporal_patch_size * patch_size^2
            flat_dim = pixel_values.shape[-1]
            patch_size = int(getattr(ip, "patch_size", 0) or 0)
            temporal_patch_size = int(getattr(ip, "temporal_patch_size", 1) or 1)
            channels = 3
            block = channels * temporal_patch_size * patch_size * patch_size
            if patch_size > 0 and flat_dim % block == 0:
                per_flat_min = pv_min.repeat_interleave(
                    temporal_patch_size * patch_size * patch_size
                ).repeat(flat_dim // block).view(1, flat_dim)
                per_flat_max = pv_max.repeat_interleave(
                    temporal_patch_size * patch_size * patch_size
                ).repeat(flat_dim // block).view(1, flat_dim)
                pv_min = per_flat_min
                pv_max = per_flat_max
            else:
                # Fallback: scalar clamp with tightest bounds
                pv_min = pv_min.max()
                pv_max = pv_max.min()

        return torch.max(torch.min(pixel_values, pv_max), pv_min)

    def preprocess(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise ValueError(
                "Expected a CHW RGB tensor in [0, 1], "
                f"got shape {tuple(image_tensor.shape)}"
            )
        # HF image processors keep the official PIL semantics when tensor
        # inputs are in the pre-rescale 0..255 range. Passing [0,1] tensors
        # would be rescaled a second time for processors with do_rescale=True.
        source = image_tensor * 255.0 if getattr(self.image_processor, "do_rescale", True) else image_tensor
        out = self.image_processor([source], return_tensors="pt")
        result = batch_feature_to_dict(out)

        # Clamp pixel_values to valid normalized range (fix bicubic overshoot)
        if "pixel_values" in result and torch.is_tensor(result["pixel_values"]):
            result["pixel_values"] = self._clamp_normalized(result["pixel_values"])

        return result
