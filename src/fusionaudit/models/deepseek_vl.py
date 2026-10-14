from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


def _ensure_importable():
    from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
    if not hasattr(MultiModalityCausalLM, "all_tied_weights_keys"):
        MultiModalityCausalLM.all_tied_weights_keys = {}

    return VLChatProcessor, MultiModalityCausalLM


@dataclass(slots=True)
class DeepSeekVLChatBackend:
    processor: Any
    model: torch.nn.Module
    tokenizer: Any


def load_deepseek_vl_chat(
    model_id: str,
    device: str = "cuda:0",
    revision: str | None = None,
) -> DeepSeekVLChatBackend:
    VLChatProcessor, MultiModalityCausalLM = _ensure_importable()
    if not hasattr(MultiModalityCausalLM, "all_tied_weights_keys"):
        MultiModalityCausalLM.all_tied_weights_keys = {}

    processor = VLChatProcessor.from_pretrained(model_id, revision=revision)
    model = MultiModalityCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to(device).eval()
    return DeepSeekVLChatBackend(processor=processor, model=model, tokenizer=processor.tokenizer)


def build_single_turn_inputs(
    backend: DeepSeekVLChatBackend,
    image: Image.Image,
    question: str,
    *,
    answer: str | None = None,
) -> Any:
    conversations = [
        {
            "role": "User",
            "content": f"<image_placeholder>{question}",
        },
        {
            "role": "Assistant",
            "content": "" if answer is None else answer,
        },
    ]
    prepare_inputs = backend.processor(
        conversations=conversations,
        images=[image.convert("RGB")],
        force_batchify=True,
    )
    return prepare_inputs.to(next(backend.model.parameters()).device)


def generate_answer(
    backend: DeepSeekVLChatBackend,
    prepare_inputs: Any,
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> str:
    inputs_embeds = backend.model.prepare_inputs_embeds(
        input_ids=prepare_inputs.input_ids,
        pixel_values=prepare_inputs.pixel_values,
        images_seq_mask=prepare_inputs.images_seq_mask,
        images_emb_mask=prepare_inputs.images_emb_mask,
    )

    outputs = backend.model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=backend.tokenizer.eos_token_id,
        bos_token_id=backend.tokenizer.bos_token_id,
        eos_token_id=backend.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
    )
    return backend.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True).strip()
