"""Differentiable image processors for original-image-space attacks."""

from fusionaudit.models.processors.base import (
    DifferentiableProcessor,
    DiffProcessorInfo,
    bounded_uint8_tensor,
    image_to_float_tensor,
    project_linf,
    tensor_to_pil,
)
from fusionaudit.models.processors.registry import get_diff_processor

__all__ = [
    "DifferentiableProcessor",
    "DiffProcessorInfo",
    "bounded_uint8_tensor",
    "get_diff_processor",
    "image_to_float_tensor",
    "project_linf",
    "tensor_to_pil",
]
