"""The 18 audited models, their nativeness levels, and the audit conditions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    short: str
    level: int  # 1: alignment-only, 2: shallow fusion, 3: deep-stack / early fusion
    env: str = "main"  # "legacy": runs with transformers 4.32 (docs/ENVIRONMENTS.md)
    revision: str | None = None  # pinned Hugging Face commit


# Table 1 order.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "SmolVLM2-2.2B", 1,
              revision="482adb537c021c86670beed01cd58990d01e72e4"),
    ModelSpec("Salesforce/blip2-opt-2.7b", "BLIP2-OPT-2.7B", 1,
              revision="59a1ef6c1e5117b3f65523d1c6066825bcf315e3"),
    ModelSpec("Salesforce/instructblip-vicuna-7b", "InstructBLIP-Vicuna-7B", 1,
              revision="19103d0c5b5263c8a7891012e08573439fb6607f"),
    ModelSpec("llava-hf/llava-1.5-7b-hf", "LLaVA-1.5-7B", 1,
              revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9"),
    ModelSpec("Qwen/Qwen-VL-Chat", "Qwen-VL-Chat", 2, env="legacy",
              revision="f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"),
    ModelSpec("Qwen/Qwen2-VL-2B-Instruct", "Qwen2-VL-2B", 2,
              revision="895c3a49bc3fa70a340399125c650a463535e71c"),
    ModelSpec("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-3B", 2,
              revision="66285546d2b821cf421d4f5eb2576359d3770cd3"),
    ModelSpec("google/gemma-3-4b-it", "Gemma3-4B", 2,
              revision="093f9f388b31de276ce2de164bdc2081324b9767"),
    ModelSpec("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B", 2,
              revision="cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    ModelSpec("google/gemma-3-12b-it", "Gemma3-12B", 2,
              revision="96b6f1eccf38110c56df3a15bffe176da04bfd80"),
    ModelSpec("deepseek-ai/deepseek-vl-7b-chat", "DeepSeek-VL-7B", 2, env="legacy",
              revision="6f16f00805f45b5249f709ce21820122eeb43556"),
    ModelSpec("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-2B", 3,
              revision="89644892e4d85e24eaac8bacfd4f463576704203"),
    ModelSpec("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL-4B", 3,
              revision="ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    ModelSpec("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-8B", 3,
              revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    ModelSpec("Qwen/Qwen3.5-2B", "Qwen3.5-2B", 3,
              revision="15852e8c16360a2fea060d615a32b45270f8a8fc"),
    ModelSpec("Qwen/Qwen3.5-4B", "Qwen3.5-4B", 3,
              revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    ModelSpec("Qwen/Qwen3.5-9B", "Qwen3.5-9B", 3,
              revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ModelSpec("google/gemma-4-E4B-it", "Gemma4-E4B", 3,
              revision="ee0ef6023621cff504d758262d4e04895a5af4a2"),
)
MODEL_BY_ID = {m.model_id: m for m in MODELS}

# Audit conditions. Pixel attacks are run only on each model's clean-correct examples.
CLEAN = "clean"
PIXEL_ATTACKS = ("vis_pgd", "end_pgd", "mantis")
TEXT_ATTACKS = ("typo_evid", "nat_evid")
ATTACKS = PIXEL_ATTACKS + TEXT_ATTACKS
BASELINE_SUITE = ("vis_pgd", "end_pgd", "typo_evid")  # A^0
STRENGTHENED_SUITE = ATTACKS  # A^+ = A^0 plus MANTIS and Nat-Evid
ATTACK_LABELS = {
    "vis_pgd": "Vis-PGD", "end_pgd": "End-PGD", "mantis": "MANTIS",
    "typo_evid": "Typo-Evid", "nat_evid": "Nat-Evid",
}


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")
