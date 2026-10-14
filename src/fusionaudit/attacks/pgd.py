"""Original-image-space PGD attacks for VLMs.

These routines
optimize an RGB image tensor before model preprocessing. The L-infinity
projection is therefore enforced on the original image pixels, and the final
artifact can be exported as a uint8 image whose per-channel pixel delta is
bounded by the requested integer budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from PIL import Image

import torch.nn.functional as F

from fusionaudit.attacks.vision import (
    _clear_rotary_cache,
    _extract_features,
    _forward_smol_image_features,
    _forward_vit,
    find_vision_encoder,
)
from fusionaudit.models.processors import get_diff_processor
from fusionaudit.models.processors.base import (
    IMAGE_INPUT_KEYS,
    bounded_uint8_tensor,
    image_to_float_tensor,
    image_to_uint8_tensor,
    move_to_device,
    project_linf,
    random_start,
    tensor_to_pil,
)


@dataclass(slots=True)
class ImageSpacePGDResult:
    adv_image: Image.Image
    adv_uint8: torch.Tensor
    adv_float: torch.Tensor
    diagnostics: dict[str, Any]


class _GradCheckpoint:
    """Context manager: enable model gradient checkpointing during PGD."""

    def __init__(self, target: Any):
        self.target = target
        self.was_off = target is not None and not getattr(target, "gradient_checkpointing", False)
        self.enabled = False

    def __enter__(self) -> bool:
        if (
            self.was_off
            and self.target is not None
            and hasattr(self.target, "gradient_checkpointing_enable")
        ):
            try:
                self.target.gradient_checkpointing_enable()
                self.enabled = True
            except Exception:
                self.enabled = False
        return self.enabled

    def __exit__(self, *exc) -> None:
        if self.enabled and hasattr(self.target, "gradient_checkpointing_disable"):
            try:
                self.target.gradient_checkpointing_disable()
            except Exception:
                pass


def _upsample_delta_to_original(
    original_image: Image.Image,
    small_adv_image: Image.Image,
    small_clean_image: Image.Image,
    eps: float,
    device: torch.device | str,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor, int]:
    """Upsample the adversarial perturbation back to original resolution.

    Takes a small adversarial image produced from a resized copy of the original.
    Computes delta on the resized grid, upsamples via bilinear interpolation to
    the original resolution, re-projects to the L-inf ball, and adds it to the
    *untouched* original image. Returns a PIL the size of the original.

    Returns: (final_pil, adv_f01_orig, adv_u8_orig, max_u8_delta)
    """
    orig_f01 = image_to_float_tensor(original_image, device=device)
    orig_u8 = image_to_uint8_tensor(original_image, device=device)

    small_f01 = image_to_float_tensor(small_clean_image, device=device)
    small_adv_f01 = image_to_float_tensor(small_adv_image, device=device)
    small_delta = small_adv_f01 - small_f01  # [3, Hs, Ws], within [-eps, eps]

    H0, W0 = orig_f01.shape[-2], orig_f01.shape[-1]
    orig_delta = F.interpolate(
        small_delta.unsqueeze(0),
        size=(H0, W0),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    orig_delta = torch.clamp(orig_delta, -eps, eps)

    adv_f01 = torch.clamp(orig_f01 + orig_delta, 0.0, 1.0)
    eps_int = round(eps * 255)
    adv_u8 = bounded_uint8_tensor(adv_f01, orig_u8, eps_int=eps_int)
    max_delta = int(
        (adv_u8.to(torch.int16) - orig_u8.to(torch.int16)).abs().max().detach().cpu().item()
    )
    final_pil = tensor_to_pil(adv_u8)
    return final_pil, adv_f01.detach().cpu(), adv_u8.detach().cpu(), max_delta


def image_space_pgd_with_oom_fallback(
    *,
    original_image: Image.Image,
    pgd_callable: Callable[[Image.Image], ImageSpacePGDResult],
    eps: float,
    resize_scales: tuple[float, ...] = (1.0, 0.75, 0.5),
    device: torch.device | str | None = None,
) -> ImageSpacePGDResult:
    """Run image-space PGD with OOM retry + delta-upsampling invariant.

    ``pgd_callable(image_for_attack)`` runs PGD on the supplied PIL (possibly a
    downsized copy of ``original_image``) and returns an ``ImageSpacePGDResult``.
    On OOM, the helper retries at the next scale in ``resize_scales``. After
    success at a scale < 1.0, the adversarial perturbation is upsampled back to
    original resolution and added to the untouched original image, so the model
    always sees an adversarial image at ``original_image.size``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W0, H0 = original_image.width, original_image.height
    attempted: list[float] = []
    last_exc: Exception | None = None

    for scale in resize_scales:
        attempted.append(float(scale))
        if scale >= 1.0:
            run_image = original_image
        else:
            Ws = max(1, int(W0 * scale))
            Hs = max(1, int(H0 * scale))
            run_image = original_image.resize((Ws, Hs), Image.Resampling.LANCZOS)
        try:
            result = pgd_callable(run_image)
        except torch.cuda.OutOfMemoryError as exc:
            last_exc = exc
            torch.cuda.empty_cache()
            continue

        diag = dict(result.diagnostics)
        diag["attack_resolution"] = [run_image.width, run_image.height]
        diag["model_input_resolution"] = [W0, H0]
        diag["resize_scale_used"] = float(scale)
        diag["resize_scales_attempted"] = attempted

        if scale >= 1.0:
            diag["delta_was_upsampled"] = False
            diag.setdefault("max_u8_delta", int(
                (result.adv_uint8.to(torch.int16) - image_to_uint8_tensor(original_image).to(torch.int16))
                .abs().max().detach().cpu().item()
            ))
            return ImageSpacePGDResult(
                adv_image=result.adv_image,
                adv_uint8=result.adv_uint8,
                adv_float=result.adv_float,
                diagnostics=diag,
            )

        final_pil, adv_f01_cpu, adv_u8_cpu, max_delta = _upsample_delta_to_original(
            original_image, result.adv_image, run_image, eps, device
        )
        diag["delta_was_upsampled"] = True
        diag["max_u8_delta"] = max_delta
        return ImageSpacePGDResult(
            adv_image=final_pil,
            adv_uint8=adv_u8_cpu,
            adv_float=adv_f01_cpu,
            diagnostics=diag,
        )

    raise torch.cuda.OutOfMemoryError(
        f"All resize scales OOM'd: {attempted}; last: {last_exc}"
    )


def _inputs_to_dict(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, dict):
        return dict(inputs)
    if hasattr(inputs, "items"):
        return dict(inputs.items())
    if hasattr(inputs, "__dict__"):
        return dict(vars(inputs))
    raise TypeError(f"Unsupported input container: {type(inputs)!r}")


def _merge_model_inputs(
    base_inputs: dict[str, Any],
    image_inputs: dict[str, Any],
) -> dict[str, Any]:
    merged = {
        k: v
        for k, v in base_inputs.items()
        if k not in IMAGE_INPUT_KEYS and k != "labels"
    }
    merged.update(image_inputs)
    return merged


def image_space_pgd(
    *,
    image: Image.Image,
    processor: Any,
    model_id: str,
    objective_fn: Callable[[dict[str, Any]], torch.Tensor],
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 40,
    seed: int | None = 1,
    device: torch.device | str | None = None,
    diff_processor: Any | None = None,
) -> ImageSpacePGDResult:
    """Run PGD on the original RGB image tensor.

    ``objective_fn`` receives differentiable image processor outputs and must
    return a scalar loss to maximize.
    """

    if diff_processor is None:
        diff_processor = get_diff_processor(model_id, processor)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x0 = image_to_float_tensor(image, device=device)
    orig_u8 = image_to_uint8_tensor(image, device=device)
    x_adv = random_start(x0, eps, seed)
    history: list[dict[str, float]] = []

    for step in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        image_inputs = move_to_device(diff_processor.preprocess(x_adv), device)
        loss = objective_fn(image_inputs)
        grad = torch.autograd.grad(loss, x_adv)[0]
        if step in {0, steps - 1}:
            history.append(
                {
                    "step": float(step),
                    "objective": float(loss.detach().cpu().item()),
                    "grad_norm": float(grad.detach().norm().cpu().item()),
                }
            )
        x_adv = project_linf(x_adv.detach() + alpha * grad.sign(), x0, eps)
        del image_inputs, loss, grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adv_u8 = bounded_uint8_tensor(x_adv, orig_u8, eps_int=round(eps * 255))
    max_u8_delta = int(
        (adv_u8.to(torch.int16) - orig_u8.to(torch.int16)).abs().max().detach().cpu().item()
    )
    diagnostics = {
        "attack_space": "original_rgb_uint8",
        "eps_float": eps,
        "eps_int": round(eps * 255),
        "alpha_float": alpha,
        "steps": steps,
        "seed": seed,
        "original_size": [image.width, image.height],
        "max_float_delta": float((x_adv.detach() - x0).abs().max().cpu().item()),
        "max_u8_delta": max_u8_delta,
        "processor": diff_processor.describe(),
        "history": history,
    }
    return ImageSpacePGDResult(
        adv_image=tensor_to_pil(adv_u8),
        adv_uint8=adv_u8.detach().cpu(),
        adv_float=x_adv.detach().cpu(),
        diagnostics=diagnostics,
    )


def end_pgd(
    model: torch.nn.Module,
    processor: Any,
    model_id: str,
    image: Image.Image,
    inputs_full: Any,
    prompt_len: int,
    *,
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 40,
    seed: int | None = 1,
    diff_processor: Any | None = None,
    gradient_checkpointing: bool = True,
) -> ImageSpacePGDResult:
    """End-PGD: maximize the cross-entropy of the teacher-forced reference answer."""

    _clear_rotary_cache(model)
    device = next(model.parameters()).device
    full_dict = _inputs_to_dict(inputs_full)
    labels = full_dict["input_ids"].clone()
    labels[:, :prompt_len] = -100

    base_inputs = {
        k: v
        for k, v in full_dict.items()
        if torch.is_tensor(v) and k not in {"pixel_values", "labels"}
    }
    base_inputs = move_to_device(base_inputs, device)
    labels = labels.to(device)
    model_dtype = next(model.parameters()).dtype

    def objective(image_inputs: dict[str, Any]) -> torch.Tensor:
        casted = {
            k: (v.to(model_dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in image_inputs.items()
        }
        model_inputs = _merge_model_inputs(base_inputs, casted)
        outputs = model(labels=labels, **model_inputs)
        return outputs.loss

    gc_target = model if gradient_checkpointing else None
    with _GradCheckpoint(gc_target) as gc_enabled:
        result = image_space_pgd(
            image=image,
            processor=processor,
            model_id=model_id,
            objective_fn=objective,
            eps=eps,
            alpha=alpha,
            steps=steps,
            seed=seed,
            device=device,
            diff_processor=diff_processor,
        )
    result.diagnostics["gradient_checkpointing_enabled"] = bool(gc_enabled)
    return result


def vis_pgd(
    model: torch.nn.Module,
    processor: Any,
    model_id: str,
    image: Image.Image,
    inputs: Any,
    *,
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 40,
    seed: int | None = 1,
    diff_processor: Any | None = None,
) -> ImageSpacePGDResult | None:
    """Vis-PGD: push the vision-encoder features away from the clean features.

    Minimizes cosine similarity between clean and adversarial ViT features,
    with the L-inf constraint applied in the original RGB pixel space.

    Returns None if the model has no separable vision encoder (native fusion).
    """
    _clear_rotary_cache(model)
    device = next(model.parameters()).device

    mid = model_id.lower()
    is_smol = "smolvlm" in mid
    vit, path = find_vision_encoder(model, model_id)
    if vit is None:
        return None

    # Enable gradient checkpointing on ViT
    _gc_was_off = not getattr(vit, "gradient_checkpointing", False)
    if _gc_was_off and hasattr(vit, "gradient_checkpointing_enable"):
        vit.gradient_checkpointing_enable()

    # Get clean features from official inputs (already on device)
    inputs_dict = _inputs_to_dict(inputs)
    pv_clean = inputs_dict["pixel_values"]
    with torch.no_grad():
        if is_smol:
            clean_feat = _forward_smol_image_features(model, pv_clean, inputs_dict).detach()
        else:
            clean_feat = _forward_vit(vit, pv_clean, inputs_dict, model_id).detach()

    def objective(image_inputs: dict[str, Any]) -> torch.Tensor:
        pv = image_inputs["pixel_values"]
        if is_smol:
            adv_feat = _forward_smol_image_features(model, pv, inputs_dict)
        else:
            adv_feat = _forward_vit(vit, pv, inputs_dict, model_id)
        # Minimize cosine similarity = maximize negative cosine similarity
        cos_sim = F.cosine_similarity(
            adv_feat.reshape(1, -1), clean_feat.reshape(1, -1)
        ).squeeze()
        return -cos_sim  # image_space_pgd maximizes, so negate

    result = image_space_pgd(
        image=image,
        processor=processor,
        model_id=model_id,
        objective_fn=objective,
        eps=eps,
        alpha=alpha,
        steps=steps,
        seed=seed,
        device=device,
        diff_processor=diff_processor,
    )
    result.diagnostics["gradient_checkpointing_enabled"] = bool(_gc_was_off)

    # Restore gradient checkpointing state
    if _gc_was_off and hasattr(vit, "gradient_checkpointing_disable"):
        vit.gradient_checkpointing_disable()

    return result
