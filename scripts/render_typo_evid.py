#!/usr/bin/env python3
"""Render the Typo-Evid images for the evaluation set.

    python scripts/render_typo_evid.py --manifest data/manifest.json --out data/typo_evid \
        [--image-cache data/images] [--hashes data/typo_evid_hashes.json]

Writes <out>/<sample_id>.png and <out>/typo_evid.jsonl (wrong answer and overlay
text). With --hashes, every rendered image is checked against the released hashes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fusionaudit.data.evalset import EvalSample, image_sha256, load_image  # noqa: E402
from fusionaudit.data.typo_evid import render_typo_evid  # noqa: E402


def load_samples(manifest: Path) -> list[EvalSample]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return [EvalSample(s["sample_id"], s["dataset_id"], s["source_index"], s["question"], s["expected_answer"])
            for s in data["samples"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-cache", type=Path, default=None)
    ap.add_argument("--hashes", type=Path, default=None, help="released Typo-Evid image hashes to check against")
    args = ap.parse_args()

    expected = {}
    if args.hashes is not None:
        expected = {r["sample_id"]: r["image_sha256"] for r in json.loads(args.hashes.read_text(encoding="utf-8"))}
    args.out.mkdir(parents=True, exist_ok=True)
    bad = []
    with (args.out / "typo_evid.jsonl").open("w", encoding="utf-8") as meta:
        for s in load_samples(args.manifest):
            image, wrong, text = render_typo_evid(load_image(s, args.image_cache), s.question, s.expected_answer, s.dataset_id)
            image.save(args.out / f"{s.sample_id}.png", format="PNG")
            meta.write(json.dumps({"sample_id": s.sample_id, "wrong_answer": wrong, "overlay_text": text},
                                  ensure_ascii=False) + "\n")
            if expected and expected.get(s.sample_id) != image_sha256(image):
                bad.append(s.sample_id)
    print(f"rendered Typo-Evid images to {args.out}")
    if bad:
        raise SystemExit(f"{len(bad)} images differ from {args.hashes}; first: {bad[0]}")
    if expected:
        print(f"matches {args.hashes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
