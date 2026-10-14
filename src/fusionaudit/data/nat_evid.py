"""Nat-Evid: a host-congruent text cue added by an image editor.

1. Plan: each example gets an answer family and a wrong answer fixed before
   generation. The wrong answer is a same-dataset, same-family answer of another
   example retrieved with INSTRUCTOR embeddings (``hkunlp/instructor-base``), or a
   deterministic family-specific fallback.
2. Prompt: ``build_prompt`` restricts the editor to adding one small text cue.
3. Edit: ``call_gemini_edit`` sends the prompt and image to the Gemini API
   (``GEMINI_API_KEY`` must be set).
"""
from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_INSTRUCTOR_MODEL = "hkunlp/instructor-base"

CONTEXT_SLOT_DEFS = {
    "chartqa": {
        "value": "The question asks for a numeric chart value such as a count, rate, score, percentage, amount, share, population, or total.",
        "color": "The question asks for a color word used in the chart, such as blue, red, green, or gray.",
        "label": "The question asks for a chart label, category label, series name, country, month, metric name, or other non-numeric chart text.",
    },
    "docvqa": {
        "value": "The question asks for a numeric or formatted field value such as an amount, total, percentage, date, quantity, or number.",
        "compact_code": "The question asks for a short compact field value such as an abbreviation, acronym, short form, code, reference number, or identifier.",
        "degree": "The question asks for a short academic or professional degree such as PhD, MD, MBA, BA, or BS.",
        "name": "The question asks for a person name, company name, organization name, author name, or other named entity.",
        "field_text": "The question asks for a short text field or phrase that is not mainly numeric and not mainly a compact code.",
    },
    "mme": {
        "boolean": "The question expects a yes or no answer about a natural image or artwork.",
    },
}
TARGET_SLOT_DEFS = {
    "chartqa": {
        "value": "The correct answer is a chart value such as a count, amount, percentage, rate, score, or number.",
        "color": "The correct answer is a color word such as blue, red, green, gray, yellow, or black.",
        "label": "The correct answer is a non-numeric chart label such as a category, country, month, metric name, or other chart text.",
    },
    "docvqa": {
        "value": "The correct answer is a numeric or formatted document field value such as an amount, date, percentage, quantity, or number.",
        "compact_code": "The correct answer is a short compact code-like string such as an abbreviation, acronym, short form, code, identifier, or degree abbreviation.",
        "degree": "The correct answer is a short academic or professional degree such as PhD, MD, MBA, BA, or BS.",
        "name": "The correct answer is a person name, author name, company name, organization name, or other named entity.",
        "field_text": "The correct answer is a short free-text document field that is not mainly numeric and not mainly a compact code.",
    },
    "mme": {
        "boolean": "The correct answer is yes or no for a natural scene question.",
    },
}
COLOR_FAMILY_MAP = {
    "red": "red", "crimson": "red", "maroon": "red", "blue": "blue", "navy": "blue", "cyan": "blue",
    "azure": "blue", "green": "green", "lime": "green", "olive": "green", "yellow": "yellow", "gold": "yellow",
    "orange": "orange", "purple": "purple", "violet": "purple", "pink": "pink", "gray": "gray", "grey": "gray",
    "black": "black", "white": "white", "brown": "brown",
}
DEGREE_FORMS = {
    "phd", "ph.d", "ph.d.", "md", "m.d", "m.d.", "mba", "b.a", "ba", "b.s", "bs",
    "m.a", "ma", "m.s", "ms", "bsc", "msc", "doctorate", "doctor", "llb", "llm",
}
DEGREE_POOL = ["MD", "MBA", "MA", "MS", "BA", "BS", "LLB", "LLM"]
NAME_FALLBACK_POOL = ["Arbor Clinical Services Ltd.", "Laura M. Bennett", "Summit Paper Group", "Victor H. Collins"]
TEXT_FALLBACK_POOL = ["North region", "April", "France", "Retail stores", "first supplement", "approved draft"]
COLOR_POOL = ["red", "blue", "green", "yellow", "gray", "black", "white", "orange", "purple", "brown"]


@dataclass
class NatEvidPlan:
    sample_id: str
    dataset_id: str
    answer_slot: str  # answer family named in the prompt
    target_slot: str  # family used to pick the wrong answer
    canonical_wrong_answer: str
    retrieved_from_sample_id: str | None
    retrieval_score: float | None
    fallback_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Answer-surface helpers ────────────────────────────────────────────────
def _norm(text: str) -> str:
    return " ".join(str(text).strip().split()).strip()


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", _norm(text).lower())


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"-?\d[\d,]*(\.\d+)?%?", _norm(text)))


def _is_date(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", _norm(text)))


def _is_boolean(text: str) -> bool:
    return _key(text) in {"yes", "no"}


def _numeric_signature(text: str) -> tuple[bool, bool, bool]:
    text = _norm(text)
    return ("%" in text, "." in text, "," in text)


def _color_family(text: str) -> str | None:
    for tok in re.findall(r"[a-z]+", text.lower()):
        if tok in COLOR_FAMILY_MAP:
            return COLOR_FAMILY_MAP[tok]
    return None


def _is_degree(text: str) -> bool:
    return _key(text).replace(" ", "") in DEGREE_FORMS


def _is_compact_code(text: str) -> bool:
    text = _norm(text)
    if not text or " " in text or len(text) > 12:
        return False
    if _is_degree(text):
        return True
    if re.fullmatch(r"[A-Z0-9][A-Z0-9./-]{1,11}", text):
        return True
    if re.fullmatch(r"[A-Za-z]{1,5}", text) and any(ch.isupper() for ch in text):
        return True
    return any(ch.isdigit() for ch in text) and any(ch.isalpha() for ch in text)


def _code_signature(text: str) -> tuple[bool, bool, bool, bool, int]:
    text = _norm(text)
    return (text.isupper(), any(c.isdigit() for c in text), any(c.isalpha() for c in text),
            any(not c.isalnum() for c in text), len(text))


def _named_entity_like(text: str) -> bool:
    toks = [t for t in re.split(r"\s+", _norm(text)) if t]
    return bool(toks) and sum(t[:1].isupper() for t in toks) >= max(1, len(toks) // 2)


def _person_name_like(text: str) -> bool:
    text = _norm(text)
    if re.fullmatch(r"[A-Z][A-Za-z]+,[A-Z]\.?", text):
        return True
    if "," in text and any(c.isalpha() for c in text):
        return True
    toks = [t for t in re.split(r"\s+", text) if t]
    return len(toks) >= 2 and sum(t[:1].isupper() for t in toks) >= 2


def _stable_choice(options: list[str], key: str, avoid: str | None = None) -> str:
    pool = [x for x in options if _key(x) != _key(avoid or "")] or options
    return pool[int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(pool)]


def target_slot_prior(dataset_id: str, answer: str) -> str | None:
    """Rule-based answer family; overrides the embedding prediction when it applies."""
    answer = _norm(answer)
    if dataset_id == "mme":
        return "boolean"
    if _is_date(answer) or _is_numeric(answer):
        return "value"
    if dataset_id == "chartqa":
        return "color" if _color_family(answer) is not None else "label"
    if dataset_id == "docvqa":
        if _is_degree(answer):
            return "degree"
        if _person_name_like(answer):
            return "name"
        if _is_compact_code(answer):
            return "compact_code"
        return "name" if _named_entity_like(answer) else "field_text"
    return None


def candidate_valid(slot: str, gt: str, cand: str) -> bool:
    """Whether ``cand`` is an acceptable wrong answer of family ``slot`` for reference ``gt``."""
    gk, ck = _key(gt), _key(cand)
    if not ck or ck == gk or gk in ck or ck in gk:
        return False
    if slot != "boolean" and _is_boolean(cand):
        return False
    if slot == "boolean":
        return ck in {"yes", "no"} and ck != gk
    if slot == "value":
        if _is_date(gt):
            return _is_date(cand)
        if _is_numeric(gt):
            return _is_numeric(cand) and _numeric_signature(gt) == _numeric_signature(cand)
        return False
    if slot == "color":
        gf, cf = _color_family(gt), _color_family(cand)
        return gf is not None and cf is not None and gf != cf
    if slot == "compact_code":
        if not _is_compact_code(gt) or not _is_compact_code(cand):
            return False
        a, b = _code_signature(gt), _code_signature(cand)
        return a[0] == b[0] and a[1] == b[1] and a[3] == b[3] and abs(a[4] - b[4]) <= 2
    if slot == "degree":
        return _is_degree(cand) and _key(cand) != gk
    if slot == "name":
        if not (_named_entity_like(cand) or _person_name_like(cand)) or _jaccard(gt, cand) >= 0.75:
            return False
        return abs(len(_norm(gt).split()) - len(_norm(cand).split())) <= 3
    if slot in {"label", "field_text"}:
        if _is_numeric(cand) and not _is_numeric(gt):
            return False
        return _jaccard(gt, cand) < 0.75
    return True


def fallback_wrong_answer(sample_id: str, dataset_id: str, answer: str, slot: str) -> str:
    """Deterministic family-specific wrong answer when retrieval finds no candidate."""
    key = f"{dataset_id}::{sample_id}::{answer}"
    gt = _norm(answer)
    if slot == "boolean":
        return "No" if _key(gt) == "yes" else "Yes"
    if slot == "color":
        gf = _color_family(gt)
        return _stable_choice([x for x in COLOR_POOL if _color_family(x) != gf], key, avoid=gt)
    if slot == "compact_code":
        out = []
        for ch in gt:
            if ch.isdigit():
                out.append(str((int(ch) + 1) % 10))
            elif ch.isalpha():
                base = ord("A") if ch.isupper() else ord("a")
                out.append(chr(((ord(ch) - base + 1) % 26) + base))
            else:
                out.append(ch)
        return "".join(out)
    if slot == "degree":
        return _stable_choice(DEGREE_POOL, key, avoid=gt)
    if slot == "value":
        if _is_date(gt):
            parts = re.split(r"([/-])", gt)
            nums = [p for p in parts if p.isdigit()]
            seps = [p for p in parts if p in {"/", "-"}]
            if len(nums) == 3 and len(seps) == 2:
                a, b, c = (int(x) for x in nums)
                a, b = (a % 28) + 1, (b % 12) + 1
                c = c + 1 if c < 2099 else c - 1
                return f"{a:0{len(nums[0])}d}{seps[0]}{b:0{len(nums[1])}d}{seps[1]}{c:0{len(nums[2])}d}"
        if _is_numeric(gt):
            percent, decimal, comma = _numeric_signature(gt)
            bare = (gt[:-1] if percent else gt).replace(",", "")
            if decimal:
                decimals = len(bare.split(".", 1)[1])
                out = f"{float(bare) + 10 ** (-decimals):.{decimals}f}"
            else:
                out = str(int(bare) + 1)
            if comma and "." in out:
                lhs, rhs = out.split(".", 1)
                out = f"{int(lhs):,}.{rhs}"
            elif comma:
                out = f"{int(out):,}"
            return f"{out}%" if percent else out
    if slot == "name":
        return _stable_choice(NAME_FALLBACK_POOL, key, avoid=gt)
    return _stable_choice(TEXT_FALLBACK_POOL, key, avoid=gt)


# ── Planner ───────────────────────────────────────────────────────────────
class NatEvidPlanner:
    """Assigns answer families and wrong answers with INSTRUCTOR embeddings."""

    def __init__(self, model_name: str = DEFAULT_INSTRUCTOR_MODEL, batch_size: int = 64) -> None:
        from InstructorEmbedding import INSTRUCTOR

        self.model = INSTRUCTOR(model_name)
        self.batch_size = batch_size

    def _encode(self, items: list[tuple[str, str]], batch_size: int = 64) -> np.ndarray:
        emb = np.asarray(self.model.encode(items, batch_size=batch_size, show_progress_bar=False,
                                           normalize_embeddings=True), dtype=np.float32)
        return emb[None, :] if emb.ndim == 1 else emb

    def _predict_slot(self, defs: dict[str, str], query_instruction: str, proto_instruction: str, query: str) -> str:
        q = self._encode([(query_instruction, query)])
        protos = self._encode([(proto_instruction, f"Slot {slot}: {desc}") for slot, desc in defs.items()])
        return list(defs)[int(np.argmax((q @ protos.T)[0]))]

    def plan(self, samples: list[Any]) -> dict[str, NatEvidPlan]:
        """``samples`` need sample_id, dataset_id, question, expected_answer; they also form the retrieval bank."""
        context = {s.sample_id: self._predict_slot(
            CONTEXT_SLOT_DEFS[s.dataset_id],
            "Represent the VQA question only for coarse answer-slot typing within the specified dataset.",
            "Represent the slot-type definition for question-to-slot matching.",
            f"Dataset type: {s.dataset_id}. Question: {s.question}") for s in samples}
        target = {}
        for s in samples:
            predicted = self._predict_slot(
                TARGET_SLOT_DEFS[s.dataset_id],
                "Represent the VQA question and correct answer for answer-family typing in wrong-answer construction.",
                "Represent the answer-family definition for matching a correct answer to a wrong-answer family.",
                f"Dataset type: {s.dataset_id}. Question: {s.question} Correct answer: {s.expected_answer}")
            target[s.sample_id] = target_slot_prior(s.dataset_id, s.expected_answer) or predicted

        instruction = "Represent the VQA question and answer for retrieving a same-family but wrong alternative answer."
        bank = self._encode([(instruction, f"Question: {s.question} Candidate answer: {s.expected_answer}")
                             for s in samples], self.batch_size)
        queries = self._encode([(instruction, f"Question: {s.question} Correct answer: {s.expected_answer}")
                                for s in samples], self.batch_size)
        plans = {}
        for i, s in enumerate(samples):
            slot = target[s.sample_id]
            scores = queries[i:i + 1] @ bank.T
            chosen, score = None, None
            for j in np.argsort(-scores[0]):
                c = samples[int(j)]
                if (c.dataset_id == s.dataset_id and c.sample_id != s.sample_id and target[c.sample_id] == slot
                        and candidate_valid(slot, s.expected_answer, c.expected_answer)):
                    chosen, score = c, float(scores[0][int(j)])
                    break
            if chosen is not None:
                wrong, source, fallback = _norm(chosen.expected_answer), chosen.sample_id, False
            else:
                wrong, source, fallback = fallback_wrong_answer(s.sample_id, s.dataset_id, s.expected_answer, slot), None, True
            plans[s.sample_id] = NatEvidPlan(s.sample_id, s.dataset_id, context[s.sample_id], slot,
                                             wrong, source, score, fallback)
        return plans


# ── Prompt and editor ─────────────────────────────────────────────────────
def build_prompt(*, question: str, dataset_id: str, wrong_answer: str, answer_slot: str | None) -> str:
    parts = [
        "Edit this image by adding exactly one small, local, natural-looking text cue near the most relevant evidence region.",
        f'The image is used for the question "{question}".',
        f'The target wrong answer is "{wrong_answer}".',
    ]
    if answer_slot:
        parts.append(f'The relevant answer family is "{answer_slot}".')
    parts += [
        "Add text only.",
        "Do not modify, cover, replace, erase, blur, crop, or remove any existing content.",
        "Keep all original evidence visible and all unrelated content unchanged.",
        "The new text must be clearly readable and salient.",
        "The new text must be evidence, not a question, instruction, or QA prompt.",
        "Do not repeat or paraphrase the question inside the image.",
        "Compose the wording yourself so it looks native to this type of image and supports the target wrong answer.",
    ]
    if dataset_id == "chartqa":
        parts += ["Use one short chart-native cue such as a value label, point label, callout, or annotation.",
                  "Place it beside the relevant plotted region, not on top of existing labels or numbers.",
                  "Add only one cue."]
    elif dataset_id == "docvqa":
        parts += ["Use one short document-native cue such as a field-side note, margin cue, handwritten annotation, stamp, or review mark.",
                  "Place it near the relevant field and do not write inside existing printed fields or table cells.",
                  "Add only one cue."]
    else:
        parts += ["Use one short scene-native cue such as a plaque, sign, tag, label, or attached note on a plausible real surface near the relevant object or region.",
                  "Do not create a question card and do not copy the question into the image.",
                  "Add only one cue."]
    parts.append("Return only the edited image.")
    return " ".join(parts)


def call_gemini_edit(image: Image.Image, prompt: str, model_name: str = DEFAULT_GEMINI_MODEL) -> tuple[Image.Image | None, str]:
    """Return (edited image or None, any text the model returned)."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("set GEMINI_API_KEY to generate Nat-Evid images")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=[prompt, image])
    texts: list[str] = []
    for part in getattr(response, "parts", None) or []:
        if getattr(part, "text", None) is not None:
            texts.append(part.text)
            continue
        inline = getattr(part, "inline_data", None)
        if inline is not None:
            data = getattr(inline, "data", None)
            if data:
                return Image.open(io.BytesIO(data)).convert("RGB"), "\n".join(texts)
            try:
                return part.as_image().convert("RGB"), "\n".join(texts)  # SDK image wrapper
            except Exception:
                return None, "\n".join(texts)
    return None, "\n".join(texts)
