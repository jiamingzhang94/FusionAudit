"""Prompt formatting, input preparation and deterministic generation."""
from __future__ import annotations
import re
import time
from typing import Any
import torch
from PIL import Image
from .processors import get_diff_processor, image_to_float_tensor
from .processors.base import IMAGE_INPUT_KEYS
THINKING_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def _is_thinking_model(model_id: str) -> bool:
    lower = model_id.lower()
    return "qwen3.5" in lower or "qwen3-" in lower

def _strip_thinking(text: str) -> str:
    return THINKING_RE.sub("", text).strip()

def _prepare_chat_text(processor, question: str, *, thinking: bool,
                       with_answer: str | None = None) -> str:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
    ]
    if with_answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": with_answer}]})

    kwargs: dict[str, Any] = {"add_generation_prompt": with_answer is None}
    if hasattr(processor, "build_chat_payload"):
        return processor.build_chat_payload(messages, thinking=thinking, **kwargs)

    kwargs["tokenize"] = False
    if thinking:
        kwargs["enable_thinking"] = False
    return processor.apply_chat_template(messages, **kwargs)

def _inputs_to_dict(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, dict):
        return dict(inputs)
    if hasattr(inputs, "items"):
        return dict(inputs.items())
    data: dict[str, Any] = {}
    if hasattr(inputs, "__dict__"):
        data.update(vars(inputs))
    else:
        for key in ("input_ids", "attention_mask", "pixel_values", "images",
                     "images_seq_mask", "images_spatial_crop", "pixel_attention_mask"):
            if hasattr(inputs, key):
                data[key] = getattr(inputs, key)
    if not data:
        raise TypeError(f"Unsupported input container: {type(inputs)!r}")
    return data

def _generate_text(model, inputs: dict[str, Any], processor,
                   thinking: bool, max_new_tokens: int = 256) -> tuple[str, float]:
    input_dict = _inputs_to_dict(inputs)
    # model.generate() accepts tensors, images and strings; processor metadata
    # (SmolVLM's 'rows'/'cols', keys starting with _meta_) is left out.
    _EXCLUDE_KEYS = {"rows", "cols", "num_crops", "num_soft_tokens_per_image"}
    model_inputs = {}
    for k, v in input_dict.items():
        if str(k).startswith("_meta_") or k in _EXCLUDE_KEYS:
            continue
        if torch.is_tensor(v) or isinstance(v, (Image.Image, str)):
            model_inputs[k] = v
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - t0

    if hasattr(processor, "decode_generated"):
        gen_text = processor.decode_generated(out_ids, inputs)
        if thinking:
            gen_text = _strip_thinking(gen_text)
        return gen_text, round(elapsed, 3)

    input_len = input_dict["input_ids"].shape[-1]
    gen_text = processor.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()
    if thinking:
        gen_text = _strip_thinking(gen_text)
    return gen_text, round(elapsed, 3)


def prepare_inputs(processor, image, question, model_id, device, *, clean=False):
    """Clean inputs go through the differentiable preprocessing; attacked images through the official processor."""
    thinking = _is_thinking_model(model_id)
    text = _prepare_chat_text(processor, question, thinking=thinking)
    inputs = _inputs_to_dict(processor(images=[image], text=[text], return_tensors="pt"))
    if clean:
        diff = get_diff_processor(model_id, processor)
        pixels = diff.preprocess(image_to_float_tensor(image, device=device))
        original = inputs.get("pixel_values")
        dtype = original.dtype if torch.is_tensor(original) else None
        for key in IMAGE_INPUT_KEYS:
            if key in pixels:
                value = pixels[key]
                if torch.is_tensor(value):
                    value = value.to(device)
                    if dtype is not None and value.is_floating_point():
                        value = value.to(dtype)
                inputs[key] = value
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def generate(model, processor, image, question, model_id, device, *, clean=False):
    inputs = prepare_inputs(processor, image, question, model_id, device, clean=clean)
    return _generate_text(model, inputs, processor, _is_thinking_model(model_id), max_new_tokens=256)[0]
