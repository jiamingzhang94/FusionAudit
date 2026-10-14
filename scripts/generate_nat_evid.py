#!/usr/bin/env python3
"""Plan and generate the Nat-Evid images for the evaluation set.

    # generate the edits from the released plans (needs GEMINI_API_KEY)
    python scripts/generate_nat_evid.py generate --manifest data/manifest.json --out data/nat_evid \
        [--image-cache data/images] [--workers 8]
    # optional: recompute the plans (needs InstructorEmbedding)
    python scripts/generate_nat_evid.py plan --manifest data/manifest.json --out data/nat_evid

A plan fixes the wrong answer and answer family of each example
(<out>/plans.jsonl). ``generate`` builds the editing prompt from the plan and the
question, writes <out>/images/<sample_id>.png, and is resumable: examples with an
image on disk are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fusionaudit.data.evalset import EvalSample, load_image  # noqa: E402
from fusionaudit.data.nat_evid import DEFAULT_GEMINI_MODEL, NatEvidPlanner, build_prompt, call_gemini_edit  # noqa: E402


def load_samples(manifest: Path) -> list[EvalSample]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
    return [EvalSample(r["sample_id"], r["dataset_id"], r["source_index"], r["question"], r["expected_answer"])
            for r in rows]


def cmd_plan(args: argparse.Namespace) -> None:
    samples = load_samples(args.manifest)
    plans = NatEvidPlanner().plan(samples)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "plans.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(plans[s.sample_id].to_dict(), ensure_ascii=False) + "\n")
    print(f"planned {len(plans)} examples -> {args.out / 'plans.jsonl'}")


def cmd_generate(args: argparse.Namespace) -> None:
    samples = {s.sample_id: s for s in load_samples(args.manifest)}
    plans = [json.loads(line) for line in (args.out / "plans.jsonl").open(encoding="utf-8")]
    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    if args.limit is not None:
        plans = plans[:args.limit]
    todo = [p for p in plans if not (image_dir / f"{p['sample_id']}.png").exists()]

    def run(plan: dict) -> dict:
        s = samples[plan["sample_id"]]
        prompt = build_prompt(question=s.question, dataset_id=s.dataset_id,
                              wrong_answer=plan["canonical_wrong_answer"], answer_slot=plan["answer_slot"])
        for attempt in range(args.retries + 1):
            try:
                edited, text = call_gemini_edit(load_image(s, args.image_cache), prompt, args.model)
                if edited is not None:
                    edited.save(image_dir / f"{s.sample_id}.png", format="PNG")
                    return {"sample_id": s.sample_id, "status": "ok", "model": args.model}
                status = f"no_image: {text[:200]}"
            except Exception as exc:  # provider errors are recorded and retried
                status = f"error: {str(exc)[:200]}"
            time.sleep(2 ** attempt)
        return {"sample_id": s.sample_id, "status": status, "model": args.model}

    with (args.out / "generations.jsonl").open("a", encoding="utf-8") as log, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, p) for p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            log.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            log.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    n_ok = sum(1 for p in plans if (image_dir / f"{p['sample_id']}.png").exists())
    print(f"{n_ok}/{len(plans)} Nat-Evid images available in {image_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    g = sub.choices["generate"]
    g.add_argument("--image-cache", type=Path, default=None)
    g.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--retries", type=int, default=2)
    g.add_argument("--limit", type=int, default=None, help="only the first N examples")
    args = ap.parse_args()
    {"plan": cmd_plan, "generate": cmd_generate}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
