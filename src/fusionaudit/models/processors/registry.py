from __future__ import annotations

from typing import Any

from fusionaudit.models.processors.deepseek_vl import DeepSeekVLDiffProcessor
from fusionaudit.models.processors.hf_torch import OfficialTorchDiffProcessor
from fusionaudit.models.processors.qwen_vl_legacy import LegacyQwenVLDiffProcessor
from fusionaudit.models.processors.smolvlm import SmolVLMDiffProcessor


def get_diff_processor(model_id: str, official_processor: Any):
    """Return a differentiable image processor for a study model."""

    lower = model_id.lower()
    if "smolvlm" in lower:
        return SmolVLMDiffProcessor(model_id, official_processor)
    if lower in {"qwen/qwen-vl", "qwen/qwen-vl-chat"}:
        return LegacyQwenVLDiffProcessor(model_id, official_processor)
    if "deepseek-vl-" in lower and "deepseek-vl2" not in lower:
        return DeepSeekVLDiffProcessor(model_id, official_processor)
    if "qwen2-vl" in lower or "qwen2.5-vl" in lower:
        return OfficialTorchDiffProcessor(
            model_id,
            official_processor,
            model_family="Qwen2/Qwen2.5-VL",
            notes=(
                "Uses the official Torchvision-backed Qwen2VLImageProcessor "
                "on a differentiable RGB tensor; smart_resize and patch packing "
                "are unchanged."
            ),
        )
    if "qwen3-vl" in lower or "qwen3.5" in lower:
        return OfficialTorchDiffProcessor(
            model_id,
            official_processor,
            model_family="Qwen3/Qwen3.5-VL",
            notes=(
                "Uses the official Torchvision-backed Qwen2VLImageProcessor "
                "with Qwen3 parameters; grid metadata is produced by the official "
                "processor from the original H/W."
            ),
        )
    if "gemma-3" in lower:
        return OfficialTorchDiffProcessor(
            model_id,
            official_processor,
            model_family="Gemma3",
            notes=(
                "Uses the official Torchvision-backed Gemma3ImageProcessor. "
                "If pan-and-scan is enabled, crop layout is a fixed function of H/W."
            ),
        )
    if "gemma-4" in lower:
        return OfficialTorchDiffProcessor(
            model_id,
            official_processor,
            model_family="Gemma4",
            notes=(
                "Uses the official Torchvision-backed Gemma4ImageProcessor for "
                "aspect-ratio resize, patchify, position ids, and padding."
            ),
        )
    if "llava" in lower:
        return OfficialTorchDiffProcessor(
            model_id,
            official_processor,
            model_family="LLaVA/CLIP",
            notes=(
                "Uses the official CLIPImageProcessor on a differentiable tensor "
                "for shortest-edge resize, center crop, rescale, and normalize."
            ),
        )
    if "blip2" in lower or "instructblip" in lower:
        return OfficialTorchDiffProcessor(
            model_id,
            official_processor,
            model_family="BLIP2/InstructBLIP",
            notes=(
                "Uses the official BlipImageProcessor on a differentiable tensor "
                "for fixed-size bicubic resize, rescale, and CLIP normalization."
            ),
        )

    return OfficialTorchDiffProcessor(
        model_id,
        official_processor,
        model_family="generic HF Torchvision",
        notes="Fallback to official image_processor with differentiable tensor input.",
    )
