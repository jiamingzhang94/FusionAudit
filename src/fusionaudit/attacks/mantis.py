"""MANTIS: a first-divergence margin attack in original image space.

Let y+ be the canonical correct continuation and y- a canonical wrong
continuation derived from the reference answer. PGD maximizes

    L = log P(y-_t | x_adv, q, y-_<t) - log P(y+_t | x_adv, q, y+_<t)

at t, the first answer-token position where y+ and y- differ. The perturbation
is bounded in L-inf norm on the RGB pixels of the original image.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from fusionaudit.attacks.vision import _clear_rotary_cache
from fusionaudit.attacks.pgd import (
    ImageSpacePGDResult,
    _GradCheckpoint,
    image_space_pgd,
)
from fusionaudit.data.wrong_answers import derive_docvqa_wrong_answer, derive_wrong_answer
from fusionaudit.models.processors.base import IMAGE_INPUT_KEYS


@dataclass(slots=True)
class CandidateSpec:
    kind: str
    qidx: int
    text: str
    labels: torch.Tensor
    static_inputs: dict[str, torch.Tensor]


def _dedupe_keep_order(values: list[str], *, max_items: int = 5) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        v = " ".join(str(value).strip().split())
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) >= max_items:
            break
    return out


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", text.strip()))


def correct_variants(answer: str, clean_generated: str | None) -> list[str]:
    ans = " ".join(str(answer).strip().split())
    variants = [ans, f"{ans}.", f"The answer is {ans}."]
    if _is_number(ans):
        try:
            x = float(ans)
            if x.is_integer():
                variants.extend([str(int(x)), f"{int(x)}.0"])
        except Exception:
            pass
    if ans.lower() in {"yes", "no"}:
        variants.extend([ans.lower(), ans.upper(), ans.capitalize()])
    if clean_generated:
        clean = " ".join(clean_generated.strip().split())
        if clean and len(clean) <= 120:
            variants.append(clean)
    return _dedupe_keep_order(variants, max_items=5)


def wrong_variants(answer: str, dataset_id: str) -> list[str]:
    ans = " ".join(str(answer).strip().split())
    wrong = derive_docvqa_wrong_answer(ans) if dataset_id == "docvqa" else derive_wrong_answer(ans)
    variants = [wrong, f"{wrong}.", f"The answer is {wrong}."]
    if re.fullmatch(r"-?\d+", ans):
        x = int(ans)
        variants.extend([str(x + 1), str(x - 1), f"The answer is {x + 1}."])
    elif re.fullmatch(r"-?\d+\.\d+", ans):
        x = float(ans)
        variants.extend([
            f"{x + 1:.2f}".rstrip("0").rstrip("."),
            f"{x * 1.1:.2f}".rstrip("0").rstrip("."),
        ])
    if ans.lower() == "yes":
        variants.extend(["no", "No.", "The answer is no."])
    elif ans.lower() == "no":
        variants.extend(["yes", "Yes.", "The answer is yes."])
    return _dedupe_keep_order(variants, max_items=5)


def _sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses: list[torch.Tensor] = []
    for batch_idx in range(shift_labels.shape[0]):
        mask = shift_labels[batch_idx].ne(-100)
        if not mask.any():
            raise ValueError("No supervised answer tokens available for loss")
        active_logits = shift_logits[batch_idx, mask, :]
        active_labels = shift_labels[batch_idx, mask]
        losses.append(F.cross_entropy(active_logits, active_labels, reduction="sum"))
    return torch.stack(losses)


def _make_labels(input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    if labels.ne(-100).sum().item() == 0:
        raise ValueError("No answer tokens available for loss")
    return labels


def _mask_to_answer_index(labels: torch.Tensor, answer_index: int) -> torch.Tensor:
    keep = labels.clone()
    positions = torch.where(keep[0].ne(-100))[0]
    if len(positions) == 0:
        raise ValueError("No answer tokens available for discriminative-prefix loss")
    answer_index = max(0, min(answer_index, len(positions) - 1))
    target_pos = positions[answer_index]
    masked = torch.full_like(keep, -100)
    masked[:, target_pos] = keep[:, target_pos]
    return masked


def _nonignored_tokens(labels: torch.Tensor) -> list[int]:
    positions = torch.where(labels[0].ne(-100))[0]
    return [int(labels[0, pos].item()) for pos in positions]


def _apply_discriminative_prefix_masks(specs: list[CandidateSpec]) -> None:
    """Mask each qidx's specs to the first answer-token index where y+ and y- differ."""
    qidxs = sorted({s.qidx for s in specs})
    for qidx in qidxs:
        correct = next(s for s in specs if s.qidx == qidx and s.kind == "correct")
        wrong = next(s for s in specs if s.qidx == qidx and s.kind == "wrong")
        c_tokens = _nonignored_tokens(correct.labels)
        w_tokens = _nonignored_tokens(wrong.labels)
        diff_index = 0
        for i, (c_tok, w_tok) in enumerate(zip(c_tokens, w_tokens)):
            if c_tok != w_tok:
                diff_index = i
                break
        else:
            diff_index = min(len(c_tokens), len(w_tokens)) - 1
            diff_index = max(0, diff_index)
        for spec in specs:
            if spec.qidx == qidx:
                spec.labels = _mask_to_answer_index(spec.labels, diff_index)


def _inputs_to_dict(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, dict):
        return dict(inputs)
    if hasattr(inputs, "items"):
        return dict(inputs.items())
    if hasattr(inputs, "__dict__"):
        return dict(vars(inputs))
    raise TypeError(f"Unsupported input container: {type(inputs)!r}")


def _build_specs(
    processor,
    image: Image.Image,
    question: str,
    correct_text: str,
    wrong_text: str,
    thinking: bool,
    device: str,
    prepare_chat_text,
) -> list[CandidateSpec]:
    """Build the correct and wrong continuations, masked to the first divergence token."""
    prompt_text = prepare_chat_text(processor, question, thinking=thinking)
    prompt_inputs = processor(images=[image], text=[prompt_text], return_tensors="pt").to(device)
    prompt_len = prompt_inputs["input_ids"].shape[1]
    del prompt_inputs

    specs: list[CandidateSpec] = []
    for kind, text in (("correct", correct_text), ("wrong", wrong_text)):
        full_text = prepare_chat_text(processor, question, thinking=thinking, with_answer=text)
        inputs_full = processor(images=[image], text=[full_text], return_tensors="pt").to(device)
        labels = _make_labels(inputs_full["input_ids"], prompt_len)
        static_inputs = {
            k: v
            for k, v in _inputs_to_dict(inputs_full).items()
            if k not in ("pixel_values", "labels") and torch.is_tensor(v)
        }
        specs.append(CandidateSpec(kind, qidx=0, text=text, labels=labels, static_inputs=static_inputs))
        del inputs_full

    _apply_discriminative_prefix_masks(specs)
    return specs


def _model_with_image_inputs(
    model,
    image_inputs: dict[str, Any],
    labels: torch.Tensor,
    static_inputs: dict[str, torch.Tensor],
):
    model_dtype = next(model.parameters()).dtype
    casted_image = {
        k: (v.to(model_dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in image_inputs.items()
    }
    kwargs: dict[str, Any] = {
        k: v for k, v in static_inputs.items() if k not in IMAGE_INPUT_KEYS
    }
    kwargs.update(casted_image)
    kwargs.setdefault("use_cache", False)
    try:
        return model(labels=labels, **kwargs)
    except TypeError as exc:
        if "use_cache" not in str(exc):
            raise
        kwargs.pop("use_cache", None)
        return model(labels=labels, **kwargs)


def _logprob(
    model,
    image_inputs: dict[str, Any],
    spec: CandidateSpec,
) -> torch.Tensor:
    outputs = _model_with_image_inputs(model, image_inputs, spec.labels, spec.static_inputs)
    nll = _sequence_nll(outputs.logits, spec.labels)
    logp = -nll
    del outputs, nll
    return logp.mean()


def _margin_objective(
    model,
    image_inputs: dict[str, Any],
    specs: list[CandidateSpec],
) -> torch.Tensor:
    correct = next(s for s in specs if s.kind == "correct")
    wrong = next(s for s in specs if s.kind == "wrong")
    return _logprob(model, image_inputs, wrong) - _logprob(model, image_inputs, correct)


def mantis(
    model,
    processor,
    model_id: str,
    image: Image.Image,
    question: str,
    expected_answer: str,
    dataset_id: str,
    clean_generated_text: str | None,
    thinking: bool,
    device: str,
    prepare_chat_text,
    *,
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 40,
    seed: int | None = 1,
    diff_processor: Any | None = None,
    gradient_checkpointing: bool = True,
) -> ImageSpacePGDResult:
    """Run MANTIS on one image.

    ``prepare_chat_text`` formats the question with the model's chat template
    (``fusionaudit.models.inference._prepare_chat_text``).
    """
    _clear_rotary_cache(model)

    cvars = correct_variants(expected_answer, clean_generated_text)
    wvars = wrong_variants(expected_answer, dataset_id)
    if not cvars or not wvars:
        raise ValueError(
            f"MANTIS needs one correct and one wrong continuation "
            f"(got {len(cvars)} / {len(wvars)} for answer={expected_answer!r})"
        )
    correct_text = cvars[0]
    wrong_text = wvars[0]

    gc_target = model if gradient_checkpointing else None
    with _GradCheckpoint(gc_target) as gc_enabled:
        torch.cuda.empty_cache()
        specs = _build_specs(
            processor, image, question, correct_text, wrong_text,
            thinking, device, prepare_chat_text,
        )

        def objective(image_inputs: dict[str, Any]) -> torch.Tensor:
            return _margin_objective(model, image_inputs, specs)

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
        del specs

    result.diagnostics["correct_variants"] = [correct_text]
    result.diagnostics["wrong_variants"] = [wrong_text]
    result.diagnostics["gradient_checkpointing_enabled"] = bool(gc_enabled)
    return result
