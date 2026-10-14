"""The fixed 1500-example evaluation set (ChartQA, DocVQA, MME) and image loading.

The set is drawn with ``random.Random(42)``: 500 random ChartQA train questions,
and 500 DocVQA validation / 500 MME test questions taken from images with the
most questions first (to maximize multi-question images). Dataset revisions are
pinned so that indices refer to the same rows.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

SOURCES: dict[str, dict[str, Any]] = {
    "chartqa": {"repo": "HuggingFaceM4/ChartQA", "subset": None, "split": "train",
                "revision": "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"},
    "docvqa": {"repo": "lmms-lab/DocVQA", "subset": "DocVQA", "split": "validation",
               "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e"},
    "mme": {"repo": "lmms-lab/MME", "subset": None, "split": "test",
            "revision": "d6c9023f017b564f7b3ccccf5348166bce8fdbcd"},
}
SEED = 42
PER_DATASET = 500
# Images larger than this are downscaled (LANCZOS) before any model sees them.
MAX_EDGE = 1800
MAX_PIXELS = 2_000_000


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    dataset_id: str
    source_index: int
    question: str
    expected_answer: str


_DATASETS: dict[str, Any] = {}


def get_dataset(dataset_id: str):
    """Load a source split at its pinned revision (cached per process)."""
    if dataset_id not in _DATASETS:
        from datasets import load_dataset

        src = SOURCES[dataset_id]
        kwargs: dict[str, Any] = {"split": src["split"], "revision": src["revision"]}
        if src["subset"]:
            kwargs["name"] = src["subset"]
        _DATASETS[dataset_id] = load_dataset(src["repo"], **kwargs)
    return _DATASETS[dataset_id]


def _first_answer(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def _grouped_selection(group_keys: list[Any], n: int, rng: random.Random) -> list[int]:
    """Take questions from the images with the most questions first."""
    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, key in enumerate(group_keys):
        groups[key].append(idx)
    selected: list[int] = []
    for group in sorted(groups.values(), key=lambda g: -len(g)):
        if len(selected) >= n:
            break
        rng.shuffle(group)
        for idx in group:
            selected.append(idx)
            if len(selected) >= n:
                break
    return sorted(selected)


def build_eval_set(seed: int = SEED, per_dataset: int = PER_DATASET) -> list[EvalSample]:
    """Rebuild the evaluation set; the defaults give the 1500 examples of the paper."""
    rng = random.Random(seed)
    samples: list[EvalSample] = []

    ds = get_dataset("chartqa")
    order = list(range(len(ds)))
    rng.shuffle(order)
    questions = ds["query"] if "query" in ds.column_names else ds["question"]
    answers = ds["label"] if "label" in ds.column_names else ds["answer"]
    for idx in sorted(order[:per_dataset]):
        samples.append(EvalSample(f"chartqa_{idx:05d}", "chartqa", idx,
                                  "" if questions[idx] is None else str(questions[idx]), _first_answer(answers[idx])))

    ds = get_dataset("docvqa")
    questions, answers = ds["question"], ds["answers"]
    for idx in _grouped_selection(ds["docId"], per_dataset, rng):
        samples.append(EvalSample(f"docvqa_{idx:05d}", "docvqa", idx,
                                  "" if questions[idx] is None else str(questions[idx]), _first_answer(answers[idx])))

    ds = get_dataset("mme")
    questions, answers = ds["question"], ds["answer"]
    for idx in _grouped_selection(ds["question_id"], min(per_dataset, len(ds)), rng):
        samples.append(EvalSample(f"mme_{idx:05d}", "mme", idx,
                                  "" if questions[idx] is None else str(questions[idx]),
                                  "" if answers[idx] is None else str(answers[idx])))
    return samples


def resize_target(width: int, height: int) -> tuple[int, int] | None:
    """Return the downscaled size for oversized images, or None."""
    scale = 1.0
    if max(width, height) > MAX_EDGE:
        scale = min(scale, MAX_EDGE / max(width, height))
    if width * height > MAX_PIXELS:
        scale = min(scale, math.sqrt(MAX_PIXELS / (width * height)))
    if scale >= 1.0:
        return None
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def load_image(sample: EvalSample, cache_dir: Path | None = None) -> Image.Image:
    """The RGB image every model and attack receives (downscaled if oversized)."""
    if cache_dir is not None:
        cached = Path(cache_dir) / f"{sample.sample_id}.png"
        if cached.exists():
            return Image.open(cached).convert("RGB")
    img = get_dataset(sample.dataset_id)[sample.source_index]["image"].convert("RGB")
    target = resize_target(*img.size)
    if target is not None:
        img = img.resize(target, Image.LANCZOS)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        img.save(Path(cache_dir) / f"{sample.sample_id}.png", format="PNG")
    return img


def image_sha256(img: Image.Image) -> str:
    """Hash of the decoded pixels (mode, size, and RGB bytes), independent of file encoding."""
    h = hashlib.sha256()
    h.update(f"{img.mode}:{img.size[0]}x{img.size[1]}:".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def sample_to_dict(sample: EvalSample) -> dict[str, Any]:
    return asdict(sample)


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
