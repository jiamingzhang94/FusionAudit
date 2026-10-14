"""Vision feature and rotary-cache helpers used by RGB-space PGD."""
from __future__ import annotations
import logging
from typing import Any
import torch
logger = logging.getLogger(__name__)
_VIT_SEARCH_PATHS = [
    "transformer.visual",         # Qwen-VL legacy
    "model.transformer.visual",   # Wrapped Qwen-VL legacy
    "model.visual",              # Qwen2.5-VL, Qwen3-VL
    "model.model.visual",        # Some wrappers
    "vision",                    # DeepSeek-style wrappers
    "model.vision",              # Wrapped vision module
    "model.vision_model",        # Gemma-3
    "model.model.vision_model",  # Gemma-3 (deeper wrapper)
    "vision_model",              # SmolVLM2
    "model.vision_tower",        # LLaVA-style
]

def find_vision_encoder(model, model_id: str):
    """Find the separable vision encoder module.

    Returns (vit_module, path_string) or (None, None) if no callable vision branch exists.
    """
    for path in _VIT_SEARCH_PATHS:
        try:
            obj = model
            for attr in path.split("."):
                obj = getattr(obj, attr)
            # Verify it's a real module with parameters
            if hasattr(obj, "forward") and sum(1 for _ in obj.parameters()) > 0:
                logger.info("Found vision encoder at '%s' (%d params)",
                            path, sum(p.numel() for p in obj.parameters()))
                return obj, path
        except AttributeError:
            continue
    logger.warning("No vision encoder found for %s — treating as native", model_id)
    return None, None

def _extract_features(output):
    """Extract feature tensor from heterogeneous ViT outputs."""
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, tuple):
        if output and all(torch.is_tensor(item) for item in output):
            if len(output) == 1:
                return output[0]
            first = output[0]
            if first.ndim >= 3 and all(
                item.ndim == first.ndim and item.shape[0] == first.shape[0]
                for item in output
            ):
                try:
                    return torch.cat(list(output), dim=1)
                except RuntimeError:
                    return output[0]
        return output[0]
    return output

def _input_get(inputs: Any, key: str, default: Any = None) -> Any:
    if isinstance(inputs, dict):
        return inputs.get(key, default)
    if hasattr(inputs, key):
        return getattr(inputs, key)
    if hasattr(inputs, "__dict__"):
        return vars(inputs).get(key, default)
    return default

def _forward_vit(vit, pixel_values: torch.Tensor,
                 inputs: dict[str, Any], model_id: str) -> torch.Tensor:
    """Forward pixel_values through the ViT, handling model-specific APIs.

    Qwen VL models: ViT.forward(hidden_states, grid_thw) with 2D pixel_values.
    SmolVLM2: 5D [B, num_patches, C, H, W] — need patch_attention_mask.
    Others: ViT.forward(pixel_values=...) with 4D pixel_values.
    """
    mid = model_id.lower()

    # Qwen VL / Qwen3.5 models: vision module takes (hidden_states, grid_thw)
    if ("qwen2.5-vl" in mid or "qwen3-vl" in mid or "qwen2-vl" in mid or "qwen3.5" in mid):
        grid_thw = _input_get(inputs, "image_grid_thw")
        if grid_thw is not None:
            out = vit(pixel_values, grid_thw=grid_thw)
            return _extract_features(out)

    # SmolVLM2 / Idefics: 5D pixel_values [B, num_patches, C, H, W]
    # ViT expects 4D [B*N, C, H, W] — reshape and process in chunks to limit memory
    if pixel_values.ndim == 5:
        B, N, C, H, W = pixel_values.shape
        pv_4d = pixel_values.reshape(B * N, C, H, W)
        # Match dtype to ViT weights
        vit_dtype = next(vit.parameters()).dtype
        pv_4d = pv_4d.to(dtype=vit_dtype)
        pam = _input_get(inputs, "pixel_attention_mask")
        if pam is not None and pam.ndim == 4:
            pam = pam.reshape(B * N, pam.shape[-2], pam.shape[-1])

        # Process in chunks to avoid OOM during PGD backward pass
        chunk_size = 4  # patches per forward
        feats = []
        total = B * N
        for i in range(0, total, chunk_size):
            pv_chunk = pv_4d[i : i + chunk_size]
            pam_chunk = pam[i : i + chunk_size] if pam is not None else None
            try:
                out = vit(pixel_values=pv_chunk, patch_attention_mask=pam_chunk)
            except TypeError:
                try:
                    out = vit(pixel_values=pv_chunk)
                except TypeError:
                    out = vit(pv_chunk)
            feats.append(_extract_features(out))
        return torch.cat(feats, dim=0)  # [B*N, seq_len, hidden_dim]

    # Gemma4: 3D pixel_values (B, patches, patch_pixels), requires pixel_position_ids
    if "gemma-4" in mid or "gemma4" in mid:
        pixel_pos_ids = _input_get(inputs, "image_position_ids")
        out = vit(pixel_values=pixel_values, pixel_position_ids=pixel_pos_ids)
        return _extract_features(out)

    # Generic 4D: try pixel_values kwarg, fall back to positional
    try:
        out = vit(pixel_values=pixel_values)
    except TypeError:
        try:
            out = vit(pixel_values)
        except TypeError:
            # Some models need additional kwargs
            out = vit(hidden_states=pixel_values)
    return _extract_features(out)

def _forward_smol_image_features(model, pixel_values: torch.Tensor,
                                 inputs: dict[str, Any]) -> torch.Tensor:
    """Forward SmolVLM images through the model's official image-feature path.

    SmolVLM's `vision_model` cannot be treated like a plain ViT on the raw
    5D `pixel_values` tensor. The model first removes padded images and derives
    `patch_attention_mask` from `pixel_attention_mask` inside
    `get_image_features()`. Calling `vision_model` directly on reshaped tensors
    can explode memory and does not match the model's real image pipeline.
    """
    owner = model
    if not hasattr(owner, "get_image_features") and hasattr(owner, "model"):
        owner = owner.model
    out = owner.get_image_features(
        pixel_values=pixel_values,
        pixel_attention_mask=_input_get(inputs, "pixel_attention_mask"),
        return_dict=True,
    )
    return _extract_features(out)

def _clear_rotary_cache(model: torch.nn.Module) -> None:
    """Clear cached rotary-embedding cos/sin tensors that may have been
    computed under ``torch.inference_mode()``.  Without this, models that
    cache these tensors (e.g. Qwen-VL-Chat's custom RotaryEmbedding) will
    raise ``RuntimeError: Inference tensors cannot be saved for backward``
    when PGD tries to backpropagate through the forward pass.
    """
    for m in model.modules():
        if hasattr(m, "_rotary_pos_emb_cache"):
            m._rotary_pos_emb_cache = None
            m._seq_len_cached = 0
        # transformers built-in RotaryEmbedding variants. Some legacy
        # implementations (e.g. HF transformers 4.32 LLaMA used by
        # deepseek-vl) register cos_cached/sin_cached as buffers that forward
        # indexes unconditionally — nulling them breaks the next forward.
        # Safer path: if the module exposes ``_set_cos_sin_cache``, ask it
        # to repopulate; otherwise skip so we never leave the attribute None.
        if hasattr(m, "cos_cached") or hasattr(m, "sin_cached"):
            if hasattr(m, "_set_cos_sin_cache"):
                seq_len = getattr(m, "max_seq_len_cached", None) or getattr(
                    m, "max_position_embeddings", 4096
                )
                inv_freq = getattr(m, "inv_freq", None)
                device = inv_freq.device if inv_freq is not None else (
                    next(m.parameters(), torch.empty(0)).device
                )
                dtype = getattr(m, "cos_cached", None)
                dtype = dtype.dtype if dtype is not None else torch.float32
                try:
                    m._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)
                except Exception:
                    pass
                continue
