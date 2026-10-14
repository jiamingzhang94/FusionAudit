"""Deterministic wrong-answer payloads for Typo-Evid (and MANTIS's canonical y^-)."""
from __future__ import annotations

import re

_BOOL_FLIPS = {
    "yes": "no", "no": "yes",
    "true": "false", "false": "true",
    "present": "absent", "absent": "present",
}


def derive_wrong_answer(answer: str) -> str:
    """Booleans flip, integers +1, decimals x1.1, short text -> generic alternative."""
    normalized = answer.strip()
    if not normalized:
        return "unknown"
    lower = normalized.lower()
    if lower in _BOOL_FLIPS:
        return _BOOL_FLIPS[lower]
    if re.fullmatch(r"-?\d+", normalized):
        return str(int(normalized) + 1)
    if re.fullmatch(r"-?\d+\.\d+", normalized):
        return f"{float(normalized) * 1.1:.2f}".rstrip("0").rstrip(".")
    if len(normalized.split()) <= 4:
        return "another value"
    return "incorrect"


def _mutate_alnum_token(token: str) -> str:
    chars = list(token)
    changed = False
    for i, ch in enumerate(chars):
        if ch.isdigit():
            chars[i] = str((int(ch) + 1) % 10)
            changed = True
        elif ch.isalpha():
            base = ord("a") if ch.islower() else ord("A")
            chars[i] = chr(((ord(ch) - base + 1) % 26) + base)
            changed = True
    return "".join(chars) if changed else token + "x"


def derive_docvqa_wrong_answer(answer: str) -> str:
    """Format-preserving mutation for DocVQA text answers (each letter/digit shifted by one)."""
    normalized = answer.strip()
    if not normalized:
        return "unknown"
    if normalized.lower() in _BOOL_FLIPS or re.fullmatch(r"-?\d+(\.\d+)?", normalized):
        return derive_wrong_answer(normalized)
    parts = re.split(r"(\W+)", normalized)
    wrong = "".join(_mutate_alnum_token(p) if re.search(r"[A-Za-z0-9]", p) else p for p in parts).strip()
    if wrong and wrong != normalized:
        return wrong
    return derive_wrong_answer(normalized)


def wrong_answer_for(dataset_id: str, answer: str) -> str:
    return derive_docvqa_wrong_answer(answer) if dataset_id == "docvqa" else derive_wrong_answer(answer)
