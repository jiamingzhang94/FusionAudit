"""Evaluation manifest and per-sample model outputs.

Manifest (``manifest.json``, written by ``scripts/build_eval_set.py``)::

    {"name": ..., "seed": 42, "sources": {dataset_id: {repo, revision, subset, split}},
     "samples": [{"sample_id", "dataset_id", "source_index", "question",
                  "expected_answer", "image_size", "image_sha256"}, ...]}

The released ``index.json`` has the same layout without the question and answer text.

Records (``records/<model>.jsonl``), one line per model output::

    {"model_id": ..., "sample_id": ..., "condition": "clean" | attack name,
     "generated_text": ...}
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .registry import ATTACKS, CLEAN, safe_name


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset_id: str
    expected_answer: str
    question: str = ""


def load_manifest(path: Path) -> list[Sample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Sample(s["sample_id"], s["dataset_id"], s["expected_answer"], s.get("question", ""))
            for s in data["samples"]]


def records_path(records_dir: Path, model_id: str) -> Path:
    return Path(records_dir) / f"{safe_name(model_id)}.jsonl"


def load_records(records_dir: Path, model_id: str) -> dict[str, dict[str, str]]:
    """Return ``{condition: {sample_id: generated_text}}`` for one model."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    with records_path(records_dir, model_id).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["condition"] not in (CLEAN,) + ATTACKS:
                raise ValueError(f"unknown condition {r['condition']!r} for {model_id}")
            if r.get("model_id") != model_id:
                raise ValueError(f"record model does not match file: {r.get('model_id')!r}")
            if not isinstance(r["generated_text"], str):
                raise ValueError("generated_text must be a string")
            if r["sample_id"] in out[r["condition"]]:
                raise ValueError(f"duplicate record: {r['sample_id']} {r['condition']}")
            out[r["condition"]][r["sample_id"]] = r["generated_text"]
    return out


def write_records(records_dir: Path, model_id: str, rows: list[dict]) -> Path:
    path = records_path(records_dir, model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"model_id": model_id, "sample_id": r["sample_id"],
                                "condition": r["condition"], "generated_text": r["generated_text"]},
                               ensure_ascii=False) + "\n")
    return path
