from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
from typing import Any

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # transformers<4.37
    try:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

from fusionaudit.models.deepseek_vl import (
    DeepSeekVLChatBackend,
    build_single_turn_inputs as build_deepseek_vl_chat_inputs,
    generate_answer as generate_deepseek_vl_chat_answer,
    load_deepseek_vl_chat,
)


class TensorDict(dict):
    """Small BatchFeature-like mapping for custom model wrappers."""

    def to(self, device: str | torch.device):
        out = TensorDict()
        for key, value in self.items():
            if torch.is_tensor(value):
                out[key] = value.to(device)
            else:
                out[key] = value
        return out

class Blip2WrapperProcessor:
    """Wrapper for BLIP-2 / InstructBLIP processors that lack chat templates.

    These models use VQA-style prompts ('Question: ... Answer:') instead of
    multi-turn chat templates.
    """

    def __init__(self, real_processor):
        self.real_processor = real_processor
        self.tokenizer = real_processor.tokenizer
        self.image_processor = real_processor.image_processor

    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        thinking: bool = False,
    ) -> str:
        del thinking
        user = next(msg for msg in messages if msg["role"] == "user")
        question = next(
            item["text"] for item in user["content"] if item["type"] == "text"
        )
        assistant = next(
            (msg for msg in messages if msg["role"] == "assistant"), None
        )
        if assistant is not None:
            answer = next(
                item["text"]
                for item in assistant["content"]
                if item["type"] == "text"
            )
            return f"Question: {question} Answer: {answer}"
        return f"Question: {question} Answer:"

    def __call__(self, *, images, text, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        assert len(images) == 1 and len(text) == 1
        text_str = text[0]
        return self.real_processor(
            images=images[0], text=text_str, return_tensors=return_tensors
        )

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        ).strip()

def _replace_first(text: str, old: str, new: str) -> str:
    idx = text.find(old)
    if idx < 0:
        return text
    return text[:idx] + new + text[idx + len(old) :]

def _qwen_get_stop_words_ids(tokenizer):
    return [[tokenizer.im_end_id], [tokenizer.im_start_id]]

def _qwen_make_context(
    tokenizer,
    query: str,
    *,
    system: str,
    max_window_size: int,
) -> tuple[str, list[int]]:
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role: str, content: str):
        role_tokens = tokenizer.encode(role, allowed_special=set(tokenizer.IMAGE_ST))
        content_tokens = tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))
        return f"{role}\n{content}", role_tokens + nl_tokens + content_tokens

    system_text, system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

    context_tokens = system_tokens + (
        nl_tokens
        + im_start_tokens
        + _tokenize_str("user", query)[1]
        + im_end_tokens
        + nl_tokens
        + im_start_tokens
        + tokenizer.encode("assistant")
        + nl_tokens
    )
    if len(context_tokens) > max_window_size:
        context_tokens = context_tokens[-max_window_size:]
    raw_text = f"{im_start}{system_text}{im_end}\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"
    return raw_text, context_tokens

def _qwen_decode_generated(
    tokenizer,
    tokens: list[int],
    *,
    raw_text_len: int,
    context_length: int,
) -> str:
    end_idx = len(tokens)
    for idx in range(context_length, len(tokens)):
        if tokens[idx] in (tokenizer.im_start_id, tokenizer.im_end_id):
            end_idx = idx
            break
    return tokenizer.decode(tokens[:end_idx], errors="replace")[raw_text_len:].strip()

@dataclass(slots=True)
class LoadedBackend:
    processor: Any
    model: torch.nn.Module

class DeepSeekVLChatProcessor:
    def __init__(self, backend: DeepSeekVLChatBackend):
        self.backend = backend
        self.tokenizer = backend.tokenizer
        self.image_processor = backend.processor.image_processor

    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        thinking: bool = False,
    ) -> dict[str, Any]:
        del add_generation_prompt, thinking
        user = next(msg for msg in messages if msg["role"] == "user")
        question = next(item["text"] for item in user["content"] if item["type"] == "text")
        answer = None
        for msg in messages:
            if msg["role"] == "assistant":
                answer = next(item["text"] for item in msg["content"] if item["type"] == "text")
        return {"question": question, "answer": answer}

    def __call__(self, *, images, text, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        assert len(images) == 1 and len(text) == 1
        payload = text[0]
        question = payload["question"] if isinstance(payload, dict) else str(payload)
        answer = payload.get("answer") if isinstance(payload, dict) else None
        return build_deepseek_vl_chat_inputs(
            self.backend,
            images[0].convert("RGB"),
            question,
            answer=answer,
        )

    def decode_generated(self, out_ids: Any, inputs: dict[str, Any]) -> str:
        del inputs
        if isinstance(out_ids, dict):
            return str(out_ids.get("text", "")).strip()
        return str(out_ids).strip()

class DeepSeekVLChatModel(torch.nn.Module):
    def __init__(self, backend: DeepSeekVLChatBackend):
        super().__init__()
        self.backend = backend
        self.base_model = backend.model
        self.vision_model = backend.model.vision_model
        self.aligner = backend.model.aligner
        self.language_model = backend.model.language_model

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        images_seq_mask: torch.Tensor,
        images_emb_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        inputs_embeds = self.base_model.prepare_inputs_embeds(
            input_ids=input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        images_seq_mask: torch.Tensor,
        images_emb_mask: torch.Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        **kwargs,
    ):
        del kwargs
        payload = SimpleNamespace(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        text = generate_deepseek_vl_chat_answer(
            self.backend,
            payload,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return {"text": text}

class LegacyQwenVLProcessor:
    SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        chat_globals = getattr(model.base_model.chat, "__globals__", {})
        self._official_make_context = chat_globals.get("make_context", _qwen_make_context)
        self._official_decode_tokens = chat_globals.get("decode_tokens", None)
        self._official_get_stop_words_ids = chat_globals.get("get_stop_words_ids", _qwen_get_stop_words_ids)
        self.image_processor = SimpleNamespace(
            image_std=[0.26862954, 0.26130258, 0.27577711],
            do_normalize=True,
        )

    def build_chat_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        thinking: bool = False,
    ) -> dict[str, Any]:
        del thinking
        user = next(m for m in messages if m["role"] == "user")
        question = next(c["text"] for c in user["content"] if c["type"] == "text")
        answer = None
        for msg in messages:
            if msg["role"] == "assistant":
                answer = next(c["text"] for c in msg["content"] if c["type"] == "text")
        return {
            "question": question,
            "answer": answer,
            "add_generation_prompt": add_generation_prompt,
        }

    def __call__(self, *, images, text, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        assert len(images) == 1 and len(text) == 1

        payload = text[0]
        if not isinstance(payload, dict):
            raise TypeError("LegacyQwenVLProcessor expects structured chat payload")

        image: Image.Image = images[0].convert("RGB")
        pixel_values = self.model.transformer.visual.image_transform(image).unsqueeze(0)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
            image.save(fp.name)
            image_path = fp.name

        query = self.tokenizer.from_list_format(
            [{"image": image_path}, {"text": payload["question"]}]
        )
        make_context = self._official_make_context
        try:
            raw_text, context_tokens = make_context(
                self.tokenizer,
                query,
                history=None,
                system=self.SYSTEM_PROMPT,
                max_window_size=self.model.generation_config.max_window_size,
                chat_format=getattr(self.model.generation_config, "chat_format", "chatml"),
            )
        except TypeError:
            raw_text, context_tokens = make_context(
                self.tokenizer,
                query,
                system=self.SYSTEM_PROMPT,
                max_window_size=self.model.generation_config.max_window_size,
            )
        if payload["answer"] is None:
            token_ids = context_tokens
        else:
            answer_tokens = self.tokenizer.encode(
                payload["answer"], allowed_special=set(self.tokenizer.IMAGE_ST)
            )
            token_ids = context_tokens + answer_tokens + [self.tokenizer.im_end_id]

        return TensorDict(
            image=image,
            image_path=image_path,
            question=payload["question"],
            pixel_values=pixel_values,
            input_ids=torch.tensor([token_ids], dtype=torch.long),
            attention_mask=torch.ones((1, len(token_ids)), dtype=torch.long),
            stop_words_ids=self._official_get_stop_words_ids(
                getattr(self.model.generation_config, "chat_format", "chatml"),
                self.tokenizer,
            ),
            # Official `model.chat()` is not currently compatible with the
            # modern Transformers cache stack for legacy Qwen-VL code, so keep
            # the official prompt formatting but use the lower-level generate
            # path that stays under our control.
            use_chat_api=False,
            use_path_forward=True,
            _meta_raw_text_len=len(raw_text),
            _meta_context_length=len(context_tokens),
        )

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens).strip()

    def decode_generated(self, out_ids: torch.Tensor, inputs: dict[str, Any]) -> str:
        if isinstance(out_ids, dict):
            return str(out_ids.get("text", "")).strip()
        if self._official_decode_tokens is not None:
            try:
                return self._official_decode_tokens(
                    out_ids[0],
                    self.tokenizer,
                    raw_text_len=inputs["_meta_raw_text_len"],
                    context_length=inputs["_meta_context_length"],
                    chat_format=getattr(self.model.generation_config, "chat_format", "chatml"),
                    verbose=False,
                    errors="replace",
                ).strip()
            except TypeError:
                pass
        return _qwen_decode_generated(
            self.tokenizer,
            out_ids[0].tolist(),
            raw_text_len=inputs["_meta_raw_text_len"],
            context_length=inputs["_meta_context_length"],
        ).strip()

class LegacyQwenVLModel(torch.nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.transformer = base_model.transformer
        self.generation_config = base_model.generation_config
        self.base_model._supports_cache_class = False
        self._supports_cache_class = False
        self._materialize_meta_rotary()
        self._patch_transformer_helpers()
        self._patch_transformer_forward()

    def _materialize_meta_rotary(self) -> None:
        target_device = next(self.base_model.parameters()).device
        for module in self.base_model.modules():
            if module.__class__.__name__ != "RotaryEmbedding":
                logn_tensor = getattr(module, "logn_tensor", None)
                if torch.is_tensor(logn_tensor):
                    if logn_tensor.is_meta and hasattr(module, "seq_length"):
                        logn_list = [
                            math.log(i, module.seq_length) if i > module.seq_length else 1
                            for i in range(1, 32768)
                        ]
                        module.logn_tensor = torch.tensor(logn_list, device=target_device)[None, :, None, None]
                    else:
                        module.logn_tensor = logn_tensor.detach().clone().to(device=target_device)
                continue
            inv_freq = getattr(module, "inv_freq", None)
            if torch.is_tensor(inv_freq):
                if inv_freq.is_meta:
                    module.inv_freq = 1.0 / (
                        module.base
                        ** (
                            torch.arange(0, module.dim, 2, device=target_device).float()
                            / module.dim
                        )
                    )
                else:
                    module.inv_freq = inv_freq.detach().clone().to(device=target_device)
            cache = getattr(module, "_rotary_pos_emb_cache", None)
            if isinstance(cache, list) and any(torch.is_tensor(t) and t.is_meta for t in cache):
                module._rotary_pos_emb_cache = None
                module._seq_len_cached = 0

    def _patch_transformer_helpers(self) -> None:
        if hasattr(self.transformer, "get_head_mask"):
            return
        def wrapped_get_head_mask(this, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
            del is_attention_chunked
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask_expanded = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask_expanded = head_mask_expanded.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask_expanded = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(f"head_mask should have dim 1 or 2, got {head_mask.dim()}")
            return head_mask_expanded.to(dtype=next(self.base_model.parameters()).dtype)

        self.transformer.get_head_mask = MethodType(wrapped_get_head_mask, self.transformer)

    def _patch_transformer_forward(self) -> None:
        if getattr(self.transformer, "_naive_cache_patch_applied", False):
            return
        orig_forward = self.transformer.forward

        def wrapped_forward(this, *args, **kwargs):
            positional_input_ids = args[0] if len(args) >= 1 else None
            input_ids = kwargs.get("input_ids", positional_input_ids)
            inputs_embeds = kwargs.get("inputs_embeds")
            if input_ids is None and inputs_embeds is not None:
                provided_inputs_embeds = inputs_embeds
                dummy_input_ids = torch.zeros(
                    inputs_embeds.shape[:2],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                if len(args) >= 1:
                    args = (dummy_input_ids, *args[1:])
                else:
                    kwargs["input_ids"] = dummy_input_ids
                kwargs.pop("inputs_embeds", None)
                if len(args) >= 1 and "input_ids" in kwargs:
                    kwargs.pop("input_ids")

                orig_wte_forward = this.wte.forward

                def fake_wte_forward(_module, token_ids, *f_args, **f_kwargs):
                    del token_ids, f_args, f_kwargs
                    return provided_inputs_embeds

                this.wte.forward = MethodType(fake_wte_forward, this.wte)
                try:
                    return orig_forward(*args, **kwargs)
                finally:
                    this.wte.forward = orig_wte_forward

            if len(args) >= 1 and "input_ids" in kwargs:
                kwargs.pop("input_ids")
            past_key_values = kwargs.get("past_key_values")
            if past_key_values is None and len(args) >= 2:
                past_key_values = args[1]
                if past_key_values is not None and not isinstance(past_key_values, (tuple, list)):
                    if hasattr(past_key_values, "to_legacy_cache"):
                        past_key_values = past_key_values.to_legacy_cache()
                    else:
                        past_key_values = None
                    args = (args[0], past_key_values, *args[2:])
            if past_key_values is not None and not isinstance(past_key_values, (tuple, list)):
                if hasattr(past_key_values, "to_legacy_cache"):
                    kwargs["past_key_values"] = past_key_values.to_legacy_cache()
                else:
                    kwargs["past_key_values"] = None
            return orig_forward(*args, **kwargs)

        self.transformer.forward = MethodType(wrapped_forward, self.transformer)
        self.transformer._naive_cache_patch_applied = True

    def _inject_image_embeds(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        visual = self.transformer.visual
        image_embeds = visual(pixel_values.to(dtype=visual.conv1.weight.dtype))
        inputs_embeds = self.transformer.wte(input_ids)

        bos_pos = torch.where(input_ids == self.base_model.config.visual["image_start_id"])
        eos_pos = torch.where(input_ids == self.base_model.config.visual["image_start_id"] + 1)
        img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
        for idx, (i, a, b) in enumerate(img_pos):
            inputs_embeds[i, a + 1 : b] = image_embeds[idx]
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        if pixel_values is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        inputs_embeds = self._inject_image_embeds(input_ids, pixel_values)
        return self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def _manual_generate(
        self,
        *,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        stop_words_ids: list[list[int]] | None = None,
    ) -> torch.Tensor:
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        stop_ids = {token for group in (stop_words_ids or []) for token in group}
        generated: list[torch.Tensor] = []

        with torch.inference_mode():
            inputs_embeds = self._inject_image_embeds(input_ids, pixel_values)
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = outputs.past_key_values
            generated.append(next_token)

            if next_token.item() in stop_ids:
                return torch.cat([input_ids, next_token], dim=1)

            running_attention = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            current_token = next_token

            for _ in range(max_new_tokens - 1):
                outputs = self.base_model(
                    input_ids=current_token,
                    attention_mask=running_attention,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                current_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                past_key_values = outputs.past_key_values
                generated.append(current_token)
                if current_token.item() in stop_ids:
                    break
                running_attention = torch.cat(
                    [
                        running_attention,
                        torch.ones(
                            (running_attention.shape[0], 1),
                            dtype=running_attention.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )

        return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)

    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image: Image.Image | None = None,
        question: str | None = None,
        **kwargs,
    ):
        use_chat_api = bool(kwargs.pop("use_chat_api", False))
        use_path_forward = bool(kwargs.pop("use_path_forward", False))
        image_path = kwargs.pop("image_path", None)
        if use_chat_api and image is not None and question is not None:
            kwargs.pop("max_new_tokens", None)
            kwargs.pop("do_sample", None)
            for key in (
                "input_ids",
                "pixel_values",
                "attention_mask",
                "stop_words_ids",
                "_meta_raw_text_len",
                "_meta_context_length",
                "labels",
            ):
                kwargs.pop(key, None)
            kwargs["use_cache"] = False
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
                image.convert("RGB").save(fp.name)
                image_path = fp.name
            try:
                query = self.tokenizer.from_list_format(
                    [{"image": image_path}, {"text": question}]
                )
                response, _history = self.base_model.chat(
                    self.tokenizer,
                    query=query,
                    history=None,
                )
            finally:
                Path(image_path).unlink(missing_ok=True)
            return {"text": str(response).strip()}

        if input_ids is not None:
            max_new_tokens = kwargs.pop("max_new_tokens", 256)
            kwargs.pop("do_sample", None)
            stop_words_ids = kwargs.pop("stop_words_ids", None)
            kwargs.pop("use_chat_api", None)
            kwargs.pop("_meta_raw_text_len", None)
            kwargs.pop("_meta_context_length", None)
            kwargs.pop("labels", None)
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise ValueError(f"Unexpected kwargs for LegacyQwenVLModel.generate: {unexpected}")
            try:
                if use_path_forward:
                    return self.base_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        stop_words_ids=stop_words_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                        return_dict_in_generate=False,
                    )
                if pixel_values is None:
                    raise ValueError("pixel_values is required for Qwen-VL adversarial generation")
                return self._manual_generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    stop_words_ids=stop_words_ids,
                )
            finally:
                if image_path:
                    Path(image_path).unlink(missing_ok=True)

        if image is not None and question is not None:
            max_new_tokens = kwargs.pop("max_new_tokens", 256)
            kwargs.pop("do_sample", None)
            kwargs.pop("use_chat_api", None)
            for key in (
                "input_ids",
                "pixel_values",
                "attention_mask",
                "stop_words_ids",
                "_meta_raw_text_len",
                "_meta_context_length",
                "labels",
            ):
                kwargs.pop(key, None)
            kwargs["use_cache"] = False
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
                image.convert("RGB").save(fp.name)
                image_path = fp.name
            try:
                query = self.tokenizer.from_list_format(
                    [{"image": image_path}, {"text": question}]
                )
                generation_config = self.base_model.generation_config
                if getattr(generation_config, "chat_format", None) != "chatml":
                    generation_config = copy.deepcopy(generation_config)
                    generation_config.chat_format = "chatml"
                response, _history = self.base_model.chat(
                    self.tokenizer,
                    query=query,
                    history=None,
                    system=LegacyQwenVLProcessor.SYSTEM_PROMPT,
                    append_history=False,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
            finally:
                Path(image_path).unlink(missing_ok=True)
            return {"text": str(response).strip()}

        kwargs.setdefault("use_cache", False)
        if pixel_values is None:
            return self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        inputs_embeds = self._inject_image_embeds(input_ids, pixel_values)
        return self.base_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def gradient_checkpointing_enable(self):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

def load_study_backend(model_id: str, device: str, *, revision: str | None = None,
                       attn_implementation: str | None = None) -> LoadedBackend:
    """Load a VLM for study.

    Args:
        revision: Hugging Face commit to load (None: the repository default).
        attn_implementation: Optional attention implementation override.
    """
    lower = model_id.lower()
    model_path = Path(model_id)
    if "deepseek-vl-" in lower and "deepseek-vl2" not in lower:
        backend = load_deepseek_vl_chat(model_id, device=device, revision=revision)
        return LoadedBackend(
            processor=DeepSeekVLChatProcessor(backend),
            model=DeepSeekVLChatModel(backend),
        )

    is_legacy_qwen_vl = lower in {"qwen/qwen-vl", "qwen/qwen-vl-chat"} or (
        model_path.exists()
        and (model_path / "visual.py").exists()
        and (model_path / "configuration_qwen.py").exists()
    )
    if is_legacy_qwen_vl:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            device_map={"": device},
            trust_remote_code=True,
        )
        model = LegacyQwenVLModel(base_model, tokenizer).eval()
        processor = LegacyQwenVLProcessor(tokenizer, model)
        return LoadedBackend(processor=processor, model=model)

    if "blip2" in lower or "instructblip" in lower:
        # AutoProcessor returns InstructBlipVideoProcessor for InstructBLIP
        # which adds an extra video-frame dimension.  Use the explicit
        # image-only processors to keep pixel_values 4-D [B,C,H,W].
        if "instructblip" in lower:
            from transformers import InstructBlipProcessor as _IBProc
            processor = _IBProc.from_pretrained(model_id, revision=revision)
        else:
            from transformers import Blip2Processor as _B2Proc
            processor = _B2Proc.from_pretrained(model_id, revision=revision)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            device_map={"": device},
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        return LoadedBackend(
            processor=Blip2WrapperProcessor(processor),
            model=model,
        )

    if AutoModelForImageTextToText is None:
        raise ImportError(
            "This transformers version does not provide AutoModelForImageTextToText "
            "or AutoModelForVision2Seq."
        )

    extra_kwargs = {}
    if attn_implementation is not None:
        extra_kwargs["attn_implementation"] = attn_implementation

    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        device_map={"": device},
        dtype=torch.bfloat16,
        trust_remote_code=True,
        **extra_kwargs,
    ).eval()
    return LoadedBackend(processor=processor, model=model)
