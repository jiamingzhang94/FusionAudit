"""Vis-PGD, End-PGD and MANTIS on one image, with a shared RGB budget and OOM fallback."""
from __future__ import annotations

import torch

from .mantis import mantis
from .pgd import end_pgd, image_space_pgd_with_oom_fallback, vis_pgd
from ..models.inference import _inputs_to_dict, _is_thinking_model, _prepare_chat_text
from ..models.processors import get_diff_processor


def attack_image(name, model, processor, model_id, image, sample, clean_text, device,
                 *, eps=8/255, alpha=1/255, steps=40, seed=1):
    if not 0 <= eps <= 1 or alpha <= 0 or steps < 1:
        raise ValueError("Require 0 <= eps <= 1, alpha > 0, steps >= 1")
    diff = get_diff_processor(model_id, processor)
    thinking = _is_thinking_model(model_id)
    kwargs = dict(eps=eps, alpha=alpha, steps=steps, seed=seed, diff_processor=diff)

    def tokenize(img, question, *, answer=None, think=thinking):
        text = _prepare_chat_text(processor, question, thinking=think, with_answer=answer)
        inputs = _inputs_to_dict(processor(images=[img], text=[text], return_tensors="pt"))
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def run(img):
        if name == "vis_pgd":
            inputs = tokenize(img, "describe this image", think=False)
            result = vis_pgd(model, processor, model_id, img, inputs, **kwargs)
            if result is None:
                raise RuntimeError(f"No separable vision encoder for {model_id}")
            return result
        if name == "end_pgd":
            prompt = tokenize(img, sample.question)
            length = int(prompt["input_ids"].shape[1])
            del prompt
            full = tokenize(img, sample.question, answer=sample.expected_answer)
            return end_pgd(model, processor, model_id, img, full, length, **kwargs)
        if name == "mantis":
            return mantis(
                model, processor, model_id, img, sample.question, sample.expected_answer,
                sample.dataset_id, clean_text, thinking, device, _prepare_chat_text, **kwargs)
        raise ValueError(f"Unknown pixel attack: {name}")

    return image_space_pgd_with_oom_fallback(original_image=image, pgd_callable=run,
                                            eps=eps, resize_scales=(1.0, 0.75, 0.5), device=device)
