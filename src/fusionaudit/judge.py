"""The shared deterministic answer judge used for every correctness decision."""
from __future__ import annotations

import math
import re


def _normalize_number(s: str) -> str | None:
    """Parse ``s`` as a number and return a canonical string, or None."""
    s = s.strip().rstrip("%").replace(",", "")
    try:
        val = float(s)
        if not math.isfinite(val):
            return None
        if val == int(val):
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        return None


def _extract_yesno(text: str) -> str | None:
    """Extract a yes/no answer from a generation."""
    text_lower = text.strip().lower()
    words = text_lower.split()
    first_word = re.sub(r"[^a-z]", "", words[0]) if words else ""
    if first_word in ("yes", "no"):
        return first_word
    last_word = re.sub(r"[^a-z]", "", words[-1]) if words else ""
    if last_word in ("yes", "no"):
        return last_word
    m = re.search(r"\b(the answer is|answer:)\s*(yes|no)\b", text_lower)
    if m:
        return m.group(2)
    has_yes = bool(re.search(r"\byes\b", text_lower))
    has_no = bool(re.search(r"\bno\b", text_lower))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None


def is_correct(generated: str, expected: str, dataset_id: str) -> bool:
    """Return whether ``generated`` answers correctly.

    MME: a yes/no answer is extracted after punctuation stripping and compared
    case-insensitively. ChartQA/DocVQA: numeric references are compared by
    value against every number in the generation (so "5" does not match "15");
    other references use case-insensitive word-boundary containment.
    """
    if not generated or not expected:
        return False
    gen = generated.strip()
    exp = expected.strip()

    if dataset_id == "mme":
        extracted = _extract_yesno(gen)
        return extracted == exp.lower() if extracted else False

    exp_num = _normalize_number(exp)
    if exp_num is not None:
        for token in re.findall(r"[\d,]+\.?\d*", gen):
            if _normalize_number(token) == exp_num:
                return True
        return False

    return re.search(r"\b" + re.escape(exp.lower()) + r"\b", gen.lower()) is not None
