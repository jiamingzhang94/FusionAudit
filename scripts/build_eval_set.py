#!/usr/bin/env python3
"""Rebuild the 1500-example evaluation set from ChartQA, DocVQA, and MME.

    python scripts/build_eval_set.py --index data/index.json --out data/manifest.json \
        [--image-cache data/images]

Downloads the three datasets at their pinned revisions, redraws the fixed sample,
and writes the manifest (questions, reference answers, image hashes) used by every
other script. With --index, the rebuilt set is checked against the released index:
sample identifiers, source rows, and image hashes. With --image-cache, the
model-input images are also saved as PNG files for reuse.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fusionaudit.data.evalset import SEED, SOURCES, build_eval_set, image_sha256, load_image, sample_to_dict  # noqa: E402

INDEX_KEYS = ("sample_id", "dataset_id", "source_index", "image_size", "image_sha256")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=None, help="released index to check the rebuilt set against")
    ap.add_argument("--image-cache", type=Path, default=None)
    ap.add_argument("--no-hash", action="store_true", help="skip image decoding and hashing")
    args = ap.parse_args()

    samples = build_eval_set()
    rows = []
    for i, s in enumerate(samples):
        row = sample_to_dict(s)
        if not args.no_hash:
            img = load_image(s, args.image_cache)
            row["image_size"] = list(img.size)
            row["image_sha256"] = image_sha256(img)
        rows.append(row)
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(samples)}", flush=True)

    if args.index is not None:
        index = json.loads(args.index.read_text(encoding="utf-8"))["samples"]
        keys = [k for k in INDEX_KEYS if not (args.no_hash and k.startswith("image_"))]
        if len(index) != len(rows):
            raise SystemExit(f"rebuilt {len(rows)} samples, index has {len(index)}")
        bad = [r["sample_id"] for r, e in zip(rows, index) if any(r[k] != e[k] for k in keys)]
        if bad:
            raise SystemExit(f"{len(bad)} samples differ from {args.index}; first: {bad[0]}")
        print(f"matches {args.index} ({', '.join(keys)})")

    manifest = {"name": "fusionaudit-eval-1500", "seed": SEED,
                "counts": dict(Counter(r["dataset_id"] for r in rows)), "sources": SOURCES, "samples": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
